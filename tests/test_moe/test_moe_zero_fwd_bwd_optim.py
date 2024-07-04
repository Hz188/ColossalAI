from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.modeling.mixtral import EPMixtralSparseMoeBlock
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer
from tests.test_moe.moe_utils import loose_close

tokens, n_experts = 100, 4
hidden_size = 8
top_k = 2


def split_grad(grad, world_size):
    with torch.no_grad():
        grad = grad.clone().detach().flatten()
        padding_size = (world_size - grad.numel() % world_size) % world_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, [0, padding_size])
        splited_grad = grad.split(grad.numel() // world_size)
    return splited_grad


@parameterize("dtype", [torch.float16])
@parameterize("master_weights", [True])
@parameterize("stage", [1])
def run_zero_with_original_model(world_size, master_weights: bool, dtype: torch.dtype, stage: int):
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(dist.get_rank())
    plugin = MoeHybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        ep_size=dist.get_world_size(),
    )

    seed_all(20095)
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
    )

    orig_model = MixtralSparseMoeBlock(config).to(dtype).cuda()

    # ori_model = DDP(orig_model.cuda(), static_graph=True).cuda()
    ori_model = DDP(orig_model.cuda()).cuda()

    zero_model = deepcopy(orig_model).to(dtype)
    zero_model = EPMixtralSparseMoeBlock.from_native_module(zero_model, ep_group=plugin.ep_group)

    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    pg_param_list = {plugin.global_dp_group: [], plugin.moe_dp_group: []}
    for p in zero_model.parameters():
        if is_moe_tensor(p):
            pg_param_list[plugin.moe_dp_group].append(p)
        else:
            pg_param_list[plugin.global_dp_group].append(p)

    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        pg_to_param_list=pg_param_list,
        master_weights=master_weights,
        initial_scale=1,
        overlap_communication=False,
        partition_grad=True,
    )

    ori_optimizer = torch.optim.SGD(ori_model.parameters(), lr=1)

    # create
    seed_all(1453 + rank)

    for _ in range(2):
        # zero-dp forward
        input_data = torch.rand(1, tokens, hidden_size).cuda()
        zero_output, zero_logits = zero_model(input_data.to(dtype))

        # torch-ddp forward
        ori_output, ori_logits = ori_model(input_data.to(dtype))
        loose_close(zero_output, ori_output, dtype=dtype)

        # zero-dp backward
        print(f"rank={rank}, before zero backward")
        zero_optimizer.backward(zero_output.mean().float())
        print(f"rank={rank}, after zero backward")
        # torch-ddp backward
        print(f"rank={rank}, before ddp backward")
        ori_output.mean().backward()
        print(f"rank={rank}, after ddp backward")

        # check grad
        name_to_p = {n: p for n, p in ori_model.module.named_parameters()}
        print(f"rank={rank} {len(list(zero_model.named_parameters()))}")
        for n, p in zero_model.named_parameters():

            print(f"rank={rank} into get_param_grad")
            zero_grad = zero_optimizer.get_param_grad(p)
            print(f"rank={rank} out get_param_grad")
            if name_to_p[n].grad is None:
                assert zero_grad is None
                print(1111111)
                continue

            print(2222)
            loose_close(zero_grad, name_to_p[n].grad, dtype=dtype)
            print(3333)

        # zero-dp step
        print(f"rank={rank}, before zero step")
        zero_optimizer.step()
        print(f"rank={rank}, after zero step")

        # original model step
        print(f"rank={rank}, before ddp step")
        ori_optimizer.step()
        print(f"rank={rank}, after ddp step")

        # check updated param
        for n, p in zero_model.named_parameters():
            loose_close(p.data, name_to_p[n].data, dtype=dtype)
        if dist.get_rank() == 0:
            print("-" * 50)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_with_original_model(world_size=world_size)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=4)
