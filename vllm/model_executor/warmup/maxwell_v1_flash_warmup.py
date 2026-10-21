# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@instrument(span_name="Maxwell-V1-Flash sparse attention warmup")
def maxwell_v1_flash_warmup(worker: "Worker") -> None:
    """Compile the sparse attention kernels before the first request.

    Otherwise they compile on whichever data-parallel rank first sees a mixed
    batch, stalling the others at the next collective.
    """
    from vllm.models.maxwell_v1_flash.nvidia.model import (
        MaxwellV1FlashSparseAttention,
    )

    if not current_platform.is_cuda():
        return
    has_sparse = any(
        isinstance(module, MaxwellV1FlashSparseAttention)
        for module in worker.get_model().modules()
    )
    if not has_sparse:
        return

    logger.info("Warming up Maxwell-V1-Flash sparse attention kernels.")
    worker.model_runner._dummy_run(
        num_tokens=16,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )
    mxfp6 = any(
        isinstance(module, MaxwellV1FlashSparseAttention) and module.indexer.index_mxfp6
        for module in worker.get_model().modules()
    )
    if mxfp6:
        from vllm.models.maxwell_v1_flash.nvidia.ops.fused_qk_norm_rope import (
            MXFP6_TOKENS_PER_PROGRAM_STEPS,
        )

        for num_tokens in MXFP6_TOKENS_PER_PROGRAM_STEPS:
            if num_tokens <= worker.model_runner.max_num_tokens:
                worker.model_runner._dummy_run(
                    num_tokens=num_tokens,
                    skip_eplb=True,
                    is_profile=True,
                    force_attention=True,
                    create_mixed_batch=True,
                )

    from vllm.models.maxwell_v1_flash.nvidia.model import MaxwellV1FlashGate
    from vllm.models.maxwell_v1_flash.nvidia.ops.split_bf16_gemm import (
        split_bf16_gemm,
        warmup_token_counts,
    )

    gate = next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, MaxwellV1FlashGate) and module._w_hi is not None
        ),
        None,
    )
    if gate is not None:
        max_tokens = worker.vllm_config.scheduler_config.max_num_batched_tokens
        for num_tokens in warmup_token_counts(max_tokens):
            x = gate._w_hi.new_zeros(num_tokens, gate._w_hi.shape[1])
            split_bf16_gemm(x, gate._w_hi, gate._w_lo)

    _warmup_triton_moe(worker)


def _warmup_triton_moe(worker: "Worker") -> None:
    """Compile every Triton MoE variant a step size can select.

    The tile depends on the gathered token count, and Triton specializes token
    counts on divisibility by 16. Every rank derives the same steps, so the
    collectives stay matched.
    """
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_configs,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    def triton_experts(module: object) -> TritonExperts | None:
        if not isinstance(module, MoERunner):
            return None
        kernel = getattr(module._quant_method, "moe_kernel", None)
        experts = getattr(kernel, "fused_experts", None)
        return experts if isinstance(experts, TritonExperts) else None

    layer, experts = next(
        (
            (module, experts)
            for module in worker.get_model().modules()
            if (experts := triton_experts(module)) is not None
        ),
        (None, None),
    )
    if layer is None or experts is None:
        return

    w1, w2 = layer.routed_experts.w13_weight, layer.routed_experts.w2_weight
    top_k = layer.moe_config.experts_per_token
    dtype = experts.quant_config.config_name(w1.dtype)
    # Under DP + EP the experts see every rank's tokens.
    dp_size = worker.vllm_config.parallel_config.data_parallel_size
    max_tokens = worker.model_runner.max_num_tokens
    # Per-rank step sizes: powers of two and the tuned table's keys, each also
    # plus one so both divisibility cases of the gathered count come up.
    steps = {1 << i for i in range(max_tokens.bit_length())}
    steps |= {
        max(1, m // dp_size)
        for m in get_moe_configs(w2.shape[0], w2.shape[2], dtype) or ()
    }
    steps |= {n + 1 for n in steps}
    seen: set[tuple] = set()
    for num_tokens in sorted(n for n in steps if n <= max_tokens):
        gathered = num_tokens * dp_size
        config = try_get_optimal_moe_config(
            tuple(w1.shape), tuple(w2.shape), top_k, dtype, gathered
        )
        key = (*sorted(config.items()), gathered * top_k % 16 == 0)
        if key in seen:
            continue
        seen.add(key)
        worker.model_runner._dummy_run(
            num_tokens=num_tokens, skip_eplb=True, is_profile=True
        )
