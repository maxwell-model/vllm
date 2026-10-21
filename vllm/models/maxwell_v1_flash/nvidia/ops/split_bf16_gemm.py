# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""BF16 x FP32 GEMM for precision-sensitive routing.

The MoE router weights must stay fp32 (the logits feed a softmax and a small
fp32 correction bias). Each fp32 weight is split into a bf16 high part and a
bf16 residual scaled by 1/256, and both bf16 Tensor Core products accumulate
into the same fp32 registers in one kernel; two separate matmuls would round
each output to bf16 and lose the recovered precision.

Follows the approach in https://github.com/Tencent/hpc-ops.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# The residual is about 2^-8 of the weight. Must match between split and GEMM.
SPLIT_SCALE = 1.0 / 256.0


# M only bounds the masks; specializing on it (M == 1, M % 16) would compile a
# new variant for the first odd-sized step after warmup.
@triton.jit(do_not_specialize=["M"])
def _split_bf16_gemm_kernel(
    x_ptr,
    w_hi_ptr,
    w_lo_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    k_blocks = tl.cdiv(K, BLOCK_K)
    blocks_per_split = tl.cdiv(k_blocks, SPLIT_K)
    k_begin = pid_k * blocks_per_split
    k_end = tl.minimum(k_begin + blocks_per_split, k_blocks)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Block-local: the tail mask compares against what is left of K from the
    # current block, so the split's starting offset goes into the pointers.
    offs_k = tl.arange(0, BLOCK_K)
    k_start = k_begin * BLOCK_K

    x_ptrs = (
        x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk
    )
    hi_ptrs = (
        w_hi_ptr + offs_n[None, :] * stride_wn + (k_start + offs_k[:, None]) * stride_wk
    )
    lo_ptrs = (
        w_lo_ptr + offs_n[None, :] * stride_wn + (k_start + offs_k[:, None]) * stride_wk
    )

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_begin, k_end):
        k_left = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=m_mask & (offs_k[None, :] < k_left), other=0.0)
        w_mask = (offs_k[:, None] < k_left) & n_mask
        w_hi = tl.load(hi_ptrs, mask=w_mask, other=0.0)
        w_lo = tl.load(lo_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w_hi, out_dtype=tl.float32)
        acc += tl.dot(x, w_lo, out_dtype=tl.float32) * SCALE
        x_ptrs += BLOCK_K * stride_xk
        hi_ptrs += BLOCK_K * stride_wk
        lo_ptrs += BLOCK_K * stride_wk

    if SPLIT_K == 1:
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    else:
        out_ptrs = out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=m_mask & n_mask)


@triton.jit
def _split_k_reduce_kernel(
    partial_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    # Fixed summation order (not atomics) keeps expert selection deterministic.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    for s in range(1, SPLIT_K):
        acc += tl.load(partial_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


# (largest M, BLOCK_M, BLOCK_N, SPLIT_K, num_warps).
_TILE_BANDS: tuple[tuple[int | None, int, int, int, int], ...] = (
    (64, 16, 32, 4, 4),
    (256, 16, 32, 2, 4),
    (512, 32, 64, 2, 4),
    (1024, 128, 64, 4, 8),
    (2048, 128, 64, 2, 8),
    (4095, 64, 64, 1, 8),
    (None, 128, 64, 1, 8),
)


def warmup_token_counts(max_tokens: int) -> list[int]:
    """One token count per tile band a step of at most ``max_tokens`` reaches."""
    counts, low = [], 1
    for top, *_ in _TILE_BANDS:
        if low > max_tokens:
            break
        counts.append(low)
        if top is None:
            break
        low = top + 1
    return counts


def split_router_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an fp32 weight into the two bf16 halves the GEMM consumes.

    Args:
        weight: [N, K] fp32.

    Returns:
        The bf16 high part and the bf16 residual, where
        ``weight ~= high + low * SPLIT_SCALE``.

    """
    if weight.dtype != torch.float32:
        weight = weight.float()
    w_hi = weight.bfloat16()
    w_lo = ((weight - w_hi.float()) / SPLIT_SCALE).bfloat16()
    return w_hi.contiguous(), w_lo.contiguous()


def split_bf16_gemm(
    x: torch.Tensor,
    w_hi: torch.Tensor,
    w_lo: torch.Tensor,
) -> torch.Tensor:
    """Compute ``x @ (w_hi + w_lo * SPLIT_SCALE).T`` with an fp32 output.

    Args:
        x: [M, K] bf16 activations.
        w_hi: [N, K] bf16 high part from ``split_router_weight``.
        w_lo: [N, K] bf16 residual from ``split_router_weight``.

    Returns:
        [M, N] fp32 logits.

    """
    M, K = x.shape
    N = w_hi.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    # At small M the M x N grid is only a few programs; split-K fills the SMs.
    block_m, block_n, split_k, num_warps = next(
        tile for top, *tile in _TILE_BANDS if top is None or top >= M
    )
    block_n = max(16, min(block_n, triton.next_power_of_2(N)))
    target = out
    if split_k > 1:
        target = torch.empty(split_k, M, N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), split_k)
    _split_bf16_gemm_kernel[grid](
        x,
        w_hi,
        w_lo,
        target,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_hi.stride(0),
        w_hi.stride(1),
        out.stride(0),
        out.stride(1),
        SCALE=SPLIT_SCALE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        SPLIT_K=split_k,
        num_warps=num_warps,
        num_stages=3,
    )
    if split_k > 1:
        _split_k_reduce_kernel[(triton.cdiv(M * N, 1024),)](
            target, out, M * N, SPLIT_K=split_k, BLOCK=1024, num_warps=4
        )
    return out


def _split_bf16_gemm_fake(
    x: torch.Tensor,
    w_hi: torch.Tensor,
    w_lo: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(x.shape[0], w_hi.shape[0], device=x.device, dtype=torch.float32)


direct_register_custom_op(
    op_name="split_bf16_gemm",
    op_func=split_bf16_gemm,
    mutates_args=[],
    fake_impl=_split_bf16_gemm_fake,
)
