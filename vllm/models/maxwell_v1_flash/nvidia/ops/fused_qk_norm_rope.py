# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Fused Q/K RMSNorm + RoPE + paged-cache insertion kernels.

Three fusion levels, each a superset of the previous one: norm+RoPE only,
plus the main K/V cache write, plus the indexer's Q/K and index-K cache write.
"""

import torch

from vllm.models.maxwell_v1_flash.nvidia.ops.mxfp6 import (
    mxfp6_encode_store,
    mxfp8_q_encode_store,
    mxfp8_q_pad_store,
    page_layout,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    SPLIT_QK: tl.constexpr,
    IS_K: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)

    if SPLIT_QK:
        if IS_K:
            in_base = k_ptr + token * k_stride_t + head * head_dim
            out_base = k_out_ptr + token * k_out_stride_t + head * head_dim
            weight_ptr = k_weight_ptr
        else:
            in_base = q_ptr + token * q_stride_t + head * head_dim
            out_base = q_out_ptr + token * q_out_stride_t + head * head_dim
            weight_ptr = q_weight_ptr
    else:
        is_k = head >= num_q_heads
        local_head = tl.where(is_k, head - num_q_heads, head)
        if is_k:
            in_base = k_ptr + token * k_stride_t + local_head * head_dim
            out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
            weight_ptr = k_weight_ptr
        else:
            in_base = q_ptr + token * q_stride_t + local_head * head_dim
            out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim
            weight_ptr = q_weight_ptr

    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / head_dim + eps)

    # NeoX RoPE on the two halves; RMSNorm output is rounded to the storage
    # dtype before rotation, as the unfused RMSNorm -> RoPE would.
    rotary_offs = tl.arange(0, HALF_BLOCK)
    rotary_mask = rotary_offs < half_rotary
    x1 = tl.load(in_base + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_base + half_rotary + rotary_offs, mask=rotary_mask, other=0.0).to(
        tl.float32
    )
    w1 = tl.load(weight_ptr + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    w2 = tl.load(
        weight_ptr + half_rotary + rotary_offs, mask=rotary_mask, other=0.0
    ).to(tl.float32)
    x1 = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
    x2 = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_base = cos_sin_cache_ptr + position * cache_stride_p
    cos = tl.load(cache_base + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_base + half_rotary + rotary_offs,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_base + rotary_offs, x1 * cos - x2 * sin, mask=rotary_mask)
    tl.store(
        out_base + half_rotary + rotary_offs,
        x2 * cos + x1 * sin,
        mask=rotary_mask,
    )


def fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Q/K RMSNorm and full NeoX RoPE for a standard QKV projection.

    ``q`` and ``k`` may be non-contiguous last-dimension slices of one QKV
    projection. Their head elements must remain contiguous, which is true for
    ``QKVParallelLinear`` output. RMSNorm weights use direct RMSNorm semantics
    (unlike the Gemma ``weight + 1`` convention used by the gated function).
    """
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError(f"expected matching BF16/FP16 Q/K, got {q.dtype}/{k.dtype}")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("Q/K head elements must be contiguous")

    if q_out_dtype is None:
        q_out_dtype = q.dtype
    if q_out_dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
    ):
        raise ValueError(f"unsupported Q output dtype {q_out_dtype}")

    n_tokens = q.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_out_dtype, device=q.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    if n_tokens == 0:
        return q_out, k_out

    half_rotary = head_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    half_block = triton.next_power_of_2(half_rotary)
    if q_out_dtype == k.dtype:
        _fused_qk_rmsnorm_rope_kernel[(n_tokens, num_q_heads + num_kv_heads)](
            q,
            k,
            q_out,
            k_out,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            cos_sin_cache.stride(0),
            num_q_heads,
            num_kv_heads,
            head_dim,
            half_rotary,
            eps,
            SPLIT_QK=False,
            IS_K=False,
            INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
            HEAD_BLOCK=head_block,
            HALF_BLOCK=half_block,
            num_warps=max(1, head_block // 64),
            num_stages=2,
        )
        return q_out, k_out

    for num_heads, is_k in ((num_q_heads, False), (num_kv_heads, True)):
        _fused_qk_rmsnorm_rope_kernel[(n_tokens, num_heads)](
            q,
            k,
            q_out,
            k_out,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            q_out.stride(0),
            k_out.stride(0),
            cos_sin_cache.stride(0),
            num_q_heads,
            num_kv_heads,
            head_dim,
            half_rotary,
            eps,
            SPLIT_QK=True,
            IS_K=is_k,
            INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
            HEAD_BLOCK=head_block,
            HALF_BLOCK=half_block,
            num_warps=max(1, head_block // 64),
            num_stages=2,
        )
    return q_out, k_out


@triton.jit
def _fused_qk_rmsnorm_rope_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    slot_mapping_ptr,
    key_cache_ptr,
    value_cache_ptr,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    q_out_stride_t,
    cache_stride_block,
    cache_stride_offset,
    cache_stride_head,
    cache_stride_dim,
    cache_block_size,
    num_actual_tokens,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
):
    """Q/K RMSNorm+RoPE with direct paged K/V-cache insertion."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        weight_ptr = k_weight_ptr
    else:
        in_base = q_ptr + token * q_stride_t + local_head * head_dim
        weight_ptr = q_weight_ptr

    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / head_dim + eps)

    rotary_offs = tl.arange(0, HALF_BLOCK)
    rotary_mask = rotary_offs < half_rotary
    x1 = tl.load(in_base + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    x2 = tl.load(
        in_base + half_rotary + rotary_offs,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(weight_ptr + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    w2 = tl.load(
        weight_ptr + half_rotary + rotary_offs,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    x1 = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
    x2 = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int64)
    rope_base = cos_sin_cache_ptr + position * cache_stride_p
    cos = tl.load(rope_base + rotary_offs, mask=rotary_mask, other=0.0).to(tl.float32)
    sin = tl.load(
        rope_base + half_rotary + rotary_offs,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_k:
        if token < num_actual_tokens:
            slot = tl.load(slot_mapping_ptr + token)
            if slot >= 0:
                block = slot // cache_block_size
                block_offset = slot % cache_block_size
                cache_base = (
                    block * cache_stride_block
                    + block_offset * cache_stride_offset
                    + local_head * cache_stride_head
                )
                tl.store(
                    key_cache_ptr + cache_base + rotary_offs * cache_stride_dim,
                    o1,
                    mask=rotary_mask,
                )
                tl.store(
                    key_cache_ptr
                    + cache_base
                    + (half_rotary + rotary_offs) * cache_stride_dim,
                    o2,
                    mask=rotary_mask,
                )
                v_base = v_ptr + token * v_stride_t + local_head * head_dim
                v = tl.load(v_base + head_offs, mask=head_mask, other=0.0)
                tl.store(
                    value_cache_ptr + cache_base + head_offs * cache_stride_dim,
                    v,
                    mask=head_mask,
                )
    else:
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim
        tl.store(out_base + rotary_offs, o1, mask=rotary_mask)
        tl.store(
            out_base + half_rotary + rotary_offs,
            o2,
            mask=rotary_mask,
        )


def fused_qk_rmsnorm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_actual_tokens: int,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused standard-GQA Q/K normalization, RoPE, and K/V cache write.

    Only Q is materialized; rotated K is written directly to the paged cache.
    ``q_out_dtype`` can request an FP8 Q buffer, folding Q quantization into the
    norm/RoPE producer instead of adding a post-RoPE quantization launch.
    """
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise ValueError(f"expected matching BF16/FP16 Q/K, got {q.dtype}/{k.dtype}")
    if v.dtype != q.dtype:
        raise ValueError(f"expected matching V dtype, got {v.dtype}/{q.dtype}")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Q/K/V head elements must be contiguous")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError(
            f"expected matching 4-D K/V caches, got "
            f"{key_cache.shape}/{value_cache.shape}"
        )
    fp8_cache = key_cache.dtype == torch.float8_e4m3fn
    if key_cache.dtype != value_cache.dtype or (
        key_cache.dtype != q.dtype and not fp8_cache
    ):
        raise ValueError(
            f"direct cache fusion requires {q.dtype} or FP8 E4M3 cache, got "
            f"{key_cache.dtype}/{value_cache.dtype}"
        )
    if key_cache.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError(
            f"bad cache head shape {key_cache.shape[2:]}, "
            f"expected {(num_kv_heads, head_dim)}"
        )

    if q_out_dtype is None:
        q_out_dtype = q.dtype
    if q_out_dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
    ):
        raise ValueError(f"unsupported Q output dtype {q_out_dtype}")

    n_tokens = q.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_out_dtype, device=q.device
    )
    if n_tokens == 0:
        return q_out
    if num_actual_tokens > n_tokens or slot_mapping.shape[0] < num_actual_tokens:
        raise ValueError(
            f"bad actual-token count {num_actual_tokens} for T={n_tokens}, "
            f"slots={slot_mapping.shape[0]}"
        )

    half_rotary = head_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    half_block = triton.next_power_of_2(half_rotary)
    _fused_qk_rmsnorm_rope_cache_kernel[(n_tokens, num_q_heads + num_kv_heads)](
        q,
        k,
        v,
        q_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        q_out.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.shape[1],
        num_actual_tokens,
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        half_rotary,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        HALF_BLOCK=half_block,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )
    return q_out


def _fused_qk_rmsnorm_rope_cache_index_fn(
    q_ptr,
    k_ptr,
    v_ptr,
    iq_ptr,
    ik_ptr,
    q_out_ptr,
    iq_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    iq_weight_ptr,
    ik_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    slot_mapping_ptr,
    index_slot_mapping_ptr,
    key_cache_ptr,
    value_cache_ptr,
    index_k_cache_ptr,
    index_k_scale,
    iq_mx_ptr,
    iq_sf_ptr,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    iq_stride_t,
    ik_stride_t,
    q_out_stride_t,
    iq_out_stride_t,
    cache_stride_block,
    cache_stride_offset,
    cache_stride_head,
    cache_stride_dim,
    cache_block_size,
    num_actual_tokens,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_index_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    index_eps: tl.constexpr,
    INDEX_K_FP8: tl.constexpr,
    INDEX_K_MXFP6: tl.constexpr,
    INDEX_Q_MXFP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    MX_PAGE_BYTES: tl.constexpr,
    MX_CODE_BYTES: tl.constexpr,
    MX_CODE_REGION: tl.constexpr,
    MX_PAGE_TOKENS: tl.constexpr,
    n_tokens,
):
    """``_fused_qk_rmsnorm_rope_cache_kernel`` plus the BSA indexer's Q/K.

    Head slots: ``[0, num_q_heads)`` main Q, then ``num_kv_heads`` main K (with
    V), then ``num_index_heads`` index Q, then one index K, which lands in the
    flat ``[slots, head_dim]`` index cache at the indexer's own slot mapping.
    Every store reproduces the unfused path's rounding. ``HEAD_MAJOR`` swaps
    the grid axes; a program runs ``TOKENS_PER_PROGRAM`` consecutive tokens
    of its head.
    """
    if HEAD_MAJOR:
        token0 = tl.program_id(1) * TOKENS_PER_PROGRAM
        head = tl.program_id(0)
    else:
        token0 = tl.program_id(0) * TOKENS_PER_PROGRAM
        head = tl.program_id(1)
    for step in tl.static_range(TOKENS_PER_PROGRAM):
        token = token0 + step
        if (TOKENS_PER_PROGRAM == 1) or (token < n_tokens):
            is_k = (head >= num_q_heads) & (head < num_q_heads + num_kv_heads)
            is_iq = (head >= num_q_heads + num_kv_heads) & (
                head < num_q_heads + num_kv_heads + num_index_heads
            )
            is_ik = head >= num_q_heads + num_kv_heads + num_index_heads
            local_head = tl.where(
                is_k,
                head - num_q_heads,
                tl.where(is_iq, head - num_q_heads - num_kv_heads, head),
            )
            local_head = tl.where(is_ik, 0, local_head)

            if is_ik:
                in_base = ik_ptr + token * ik_stride_t
                weight_ptr = ik_weight_ptr
                norm_eps = index_eps
            elif is_iq:
                in_base = iq_ptr + token * iq_stride_t + local_head * head_dim
                weight_ptr = iq_weight_ptr
                norm_eps = index_eps
            elif is_k:
                in_base = k_ptr + token * k_stride_t + local_head * head_dim
                weight_ptr = k_weight_ptr
                norm_eps = eps
            else:
                in_base = q_ptr + token * q_stride_t + local_head * head_dim
                weight_ptr = q_weight_ptr
                norm_eps = eps

            head_offs = tl.arange(0, HEAD_BLOCK)
            head_mask = head_offs < head_dim
            x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / head_dim + norm_eps)

            rotary_offs = tl.arange(0, HALF_BLOCK)
            rotary_mask = rotary_offs < half_rotary
            x1 = tl.load(in_base + rotary_offs, mask=rotary_mask, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(
                in_base + half_rotary + rotary_offs,
                mask=rotary_mask,
                other=0.0,
            ).to(tl.float32)
            w1 = tl.load(weight_ptr + rotary_offs, mask=rotary_mask, other=0.0).to(
                tl.float32
            )
            w2 = tl.load(
                weight_ptr + half_rotary + rotary_offs,
                mask=rotary_mask,
                other=0.0,
            ).to(tl.float32)
            x1 = (x1 * inv_rms * w1).to(INPUT_DTYPE).to(tl.float32)
            x2 = (x2 * inv_rms * w2).to(INPUT_DTYPE).to(tl.float32)

            position = tl.load(positions_ptr + token).to(tl.int64)
            rope_base = cos_sin_cache_ptr + position * cache_stride_p
            cos = tl.load(rope_base + rotary_offs, mask=rotary_mask, other=0.0).to(
                tl.float32
            )
            sin = tl.load(
                rope_base + half_rotary + rotary_offs,
                mask=rotary_mask,
                other=0.0,
            ).to(tl.float32)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin

            if is_ik:
                if token < num_actual_tokens:
                    slot = tl.load(index_slot_mapping_ptr + token).to(tl.int64)
                    if INDEX_K_MXFP6:
                        if slot >= 0:
                            # Matches MBSA write_index_k on the rounded K.
                            x = tl.reshape(tl.trans(tl.join(o1, o2)), (2 * HALF_BLOCK,))
                            mxfp6_encode_store(
                                x.to(INPUT_DTYPE).to(tl.float32), slot,
                                index_k_cache_ptr, True, MX_PAGE_BYTES,
                                MX_CODE_BYTES, MX_CODE_REGION, MX_PAGE_TOKENS,
                                2 * HALF_BLOCK, 32,
                            )  # fmt: skip
                    elif slot >= 0:
                        out_base = index_k_cache_ptr + slot * head_dim
                        if INDEX_K_FP8:
                            # Scale in BF16, clamp, cast.
                            s1 = (o1.to(INPUT_DTYPE).to(tl.float32) * index_k_scale).to(
                                INPUT_DTYPE
                            )
                            s2 = (o2.to(INPUT_DTYPE).to(tl.float32) * index_k_scale).to(
                                INPUT_DTYPE
                            )
                            s1 = tl.minimum(
                                tl.maximum(s1.to(tl.float32), -FP8_MAX), FP8_MAX
                            )
                            s2 = tl.minimum(
                                tl.maximum(s2.to(tl.float32), -FP8_MAX), FP8_MAX
                            )
                            tl.store(out_base + rotary_offs, s1, mask=rotary_mask)
                            tl.store(
                                out_base + half_rotary + rotary_offs,
                                s2,
                                mask=rotary_mask,
                            )
                        else:
                            tl.store(out_base + rotary_offs, o1, mask=rotary_mask)
                            tl.store(
                                out_base + half_rotary + rotary_offs,
                                o2,
                                mask=rotary_mask,
                            )
            elif is_iq:
                out_base = iq_out_ptr + token * iq_out_stride_t + local_head * head_dim
                tl.store(out_base + rotary_offs, o1, mask=rotary_mask)
                tl.store(out_base + half_rotary + rotary_offs, o2, mask=rotary_mask)
                if INDEX_Q_MXFP8:
                    # Matches MBSA quantize_q_mxfp8_fused into
                    # [T, 2 * num_index_heads, 128]: this head's row, and the padding
                    # row num_index_heads below it.
                    x = tl.reshape(tl.trans(tl.join(o1, o2)), (2 * HALF_BLOCK,))
                    pad = 2 * num_index_heads
                    mxfp8_q_encode_store(
                        x.to(INPUT_DTYPE).to(tl.float32),
                        iq_mx_ptr + (token * pad + local_head) * head_dim,
                        iq_sf_ptr + token * pad + local_head, 2 * HALF_BLOCK, 32,
                    )  # fmt: skip
                    pad_row = token * pad + num_index_heads + local_head
                    mxfp8_q_pad_store(
                        iq_mx_ptr + pad_row * head_dim, iq_sf_ptr + pad_row,
                        2 * HALF_BLOCK,
                    )  # fmt: skip
            elif is_k:
                if key_cache_ptr.dtype.element_ty == tl.float8e4nv:
                    o1 = tl.minimum(
                        tl.maximum(o1.to(INPUT_DTYPE).to(tl.float32), -FP8_MAX), FP8_MAX
                    )
                    o2 = tl.minimum(
                        tl.maximum(o2.to(INPUT_DTYPE).to(tl.float32), -FP8_MAX), FP8_MAX
                    )
                if token < num_actual_tokens:
                    slot = tl.load(slot_mapping_ptr + token).to(tl.int64)
                    if slot >= 0:
                        block = slot // cache_block_size
                        block_offset = slot % cache_block_size
                        cache_base = (
                            block * cache_stride_block
                            + block_offset * cache_stride_offset
                            + local_head * cache_stride_head
                        )
                        tl.store(
                            key_cache_ptr + cache_base + rotary_offs * cache_stride_dim,
                            o1,
                            mask=rotary_mask,
                        )
                        tl.store(
                            key_cache_ptr
                            + cache_base
                            + (half_rotary + rotary_offs) * cache_stride_dim,
                            o2,
                            mask=rotary_mask,
                        )
                        v_base = v_ptr + token * v_stride_t + local_head * head_dim
                        v = tl.load(v_base + head_offs, mask=head_mask, other=0.0)
                        if value_cache_ptr.dtype.element_ty == tl.float8e4nv:
                            v = tl.minimum(
                                tl.maximum(v.to(tl.float32), -FP8_MAX), FP8_MAX
                            )
                        tl.store(
                            value_cache_ptr + cache_base + head_offs * cache_stride_dim,
                            v,
                            mask=head_mask,
                        )
            else:
                if q_out_ptr.dtype.element_ty == tl.float8e4nv:
                    o1 = tl.minimum(
                        tl.maximum(o1.to(INPUT_DTYPE).to(tl.float32), -FP8_MAX), FP8_MAX
                    )
                    o2 = tl.minimum(
                        tl.maximum(o2.to(INPUT_DTYPE).to(tl.float32), -FP8_MAX), FP8_MAX
                    )
                out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim
                tl.store(out_base + rotary_offs, o1, mask=rotary_mask)
                tl.store(out_base + half_rotary + rotary_offs, o2, mask=rotary_mask)


# Token counts are not specialised so warmup covers every variant serving
# hits; the one-token kernel keeps the num_actual_tokens == 1 (decode) case.
_fused_qk_rmsnorm_rope_cache_index_kernel = triton.jit(do_not_specialize=["n_tokens"])(
    _fused_qk_rmsnorm_rope_cache_index_fn
)
_fused_qk_rmsnorm_rope_cache_index_multi_kernel = triton.jit(
    do_not_specialize=["n_tokens", "num_actual_tokens"]
)(_fused_qk_rmsnorm_rope_cache_index_fn)

#: Token counts from which an MXFP6 launch gives a program 2, then 3 tokens.
MXFP6_TOKENS_PER_PROGRAM_STEPS = (256, 4096)


def mxfp6_tokens_per_program(n_tokens: int) -> int:
    """Consecutive tokens a program of the MXFP6 launch takes.

    Each value is its own compiled variant, so warmup has to reach every one.
    """
    return 1 + sum(n_tokens >= step for step in MXFP6_TOKENS_PER_PROGRAM_STEPS)


def fused_qk_rmsnorm_rope_cache_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_k_scale: float | None,
    num_actual_tokens: int,
    eps: float,
    index_eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    head_dim: int,
    q_out_dtype: torch.dtype | None = None,
    index_q_out_dtype: torch.dtype | None = None,
    index_q_mx: torch.Tensor | None = None,
    index_q_sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fused_qk_rmsnorm_rope_cache`` with the BSA indexer's Q/K in the launch.

    Index Q comes back as ``[T, num_index_heads, head_dim]``; index K is
    written into the flat ``[slots, head_dim]`` ``index_k_cache``. An E4M3
    index cache needs the indexer's ``index_k_scale``; a flat uint8 cache is
    the packed MXFP6 one, written as MBSA's ``mxfp6.write.write_index_k``
    would. The RoPE table is shared with the indexer.

    ``index_q_mx`` (``[>= T, 2 * num_index_heads, 128]`` uint8) and
    ``index_q_sf`` (``[>= T, 2 * num_index_heads]`` int32) also receive index Q
    as MBSA's ``mxfp6.write.quantize_q_mxfp8_fused`` would produce it, for the
    MXFP6 scorers.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"expected BF16/FP16 Q, got {q.dtype}")
    for name, t in (("k", k), ("v", v), ("index_q", index_q), ("index_k", index_k)):
        if t.dtype != q.dtype or t.stride(-1) != 1:
            raise ValueError(f"{name} must be {q.dtype} with contiguous head elements")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError(
            f"expected matching 4-D K/V caches, got "
            f"{key_cache.shape}/{value_cache.shape}"
        )
    if key_cache.dtype != value_cache.dtype or key_cache.dtype not in (
        q.dtype,
        torch.float8_e4m3fn,
    ):
        raise ValueError(
            f"direct cache fusion requires {q.dtype} or FP8 E4M3 cache, got "
            f"{key_cache.dtype}/{value_cache.dtype}"
        )
    if key_cache.shape[2:] != (num_kv_heads, head_dim):
        raise ValueError(
            f"bad cache head shape {key_cache.shape[2:]}, "
            f"expected {(num_kv_heads, head_dim)}"
        )
    index_k_fp8 = index_k_cache.dtype == torch.float8_e4m3fn
    index_k_mxfp6 = index_k_cache.dtype == torch.uint8
    if index_k_cache.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.uint8,
    ):
        raise ValueError(f"unsupported index-K cache dtype {index_k_cache.dtype}")
    if index_k_mxfp6:
        if index_k_cache.ndim != 1 or head_dim != 128:
            raise ValueError("the packed MXFP6 cache is flat bytes of 128-dim keys")
    elif index_k_cache.ndim != 2 or index_k_cache.stride() != (head_dim, 1):
        raise ValueError("index-K cache must be a contiguous [slots, head_dim] view")
    if index_k_fp8 and index_k_scale is None:
        raise ValueError("an E4M3 index-K cache needs the frozen index-K scale")
    if index_q.shape[-1] != num_index_heads * head_dim or index_k.shape[-1] != head_dim:
        raise ValueError("index Q/K widths do not match the index head geometry")

    q_out_dtype = q.dtype if q_out_dtype is None else q_out_dtype
    index_q_out_dtype = q.dtype if index_q_out_dtype is None else index_q_out_dtype
    for out_dtype in (q_out_dtype, index_q_out_dtype):
        if out_dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
            raise ValueError(f"unsupported Q output dtype {out_dtype}")

    n_tokens = q.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_out_dtype, device=q.device
    )
    iq_out = torch.empty(
        (n_tokens, num_index_heads, head_dim),
        dtype=index_q_out_dtype,
        device=q.device,
    )
    if n_tokens == 0:
        return q_out, iq_out
    if (
        num_actual_tokens > n_tokens
        or slot_mapping.shape[0] < num_actual_tokens
        or index_slot_mapping.shape[0] < num_actual_tokens
    ):
        raise ValueError(
            f"bad actual-token count {num_actual_tokens} for T={n_tokens}, "
            f"slots={slot_mapping.shape[0]}/{index_slot_mapping.shape[0]}"
        )

    index_q_mxfp8 = index_q_mx is not None
    if index_q_mx is not None:
        if index_q_sf is None:
            raise ValueError("index_q_mx needs its index_q_sf scales")
        iq_mx, iq_sf = index_q_mx, index_q_sf
        pad = 2 * num_index_heads
        if (
            head_dim != 128
            or iq_mx.dtype != torch.uint8
            or iq_mx.shape[0] < n_tokens
            or iq_mx.shape[1:] != (pad, head_dim)
            or iq_sf.dtype != torch.int32
            or iq_sf.shape[0] < n_tokens
            or iq_sf.shape[1:] != (pad,)
            or not (iq_mx.is_contiguous() and iq_sf.is_contiguous())
        ):
            raise ValueError("bad MXFP8 index-Q buffers for this launch")
    else:
        iq_mx = iq_sf = iq_out  # never read
    half_rotary = head_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    half_block = triton.next_power_of_2(half_rotary)
    heads = num_q_heads + num_kv_heads + num_index_heads + 1
    tokens_per_program = mxfp6_tokens_per_program(n_tokens) if index_k_mxfp6 else 1
    n_programs = triton.cdiv(n_tokens, tokens_per_program)
    # Head-major interleaves the heavy MXFP6 index heads with the main heads
    # instead of leaving them as a tail. Grid y stops at 65535 rows.
    head_major = index_k_mxfp6 and n_programs <= 65535
    layout = (
        page_layout()
        if index_k_mxfp6
        else dict.fromkeys(("PAGE_BYTES", "CODE_BYTES", "CODE_REGION", "TOKENS"), 1)
    )
    kernel = (
        _fused_qk_rmsnorm_rope_cache_index_multi_kernel
        if tokens_per_program > 1
        else _fused_qk_rmsnorm_rope_cache_index_kernel
    )
    kernel[(heads, n_programs) if head_major else (n_programs, heads)](
        q,
        k,
        v,
        index_q,
        index_k,
        q_out,
        iq_out,
        q_weight,
        k_weight,
        index_q_weight,
        index_k_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        key_cache,
        value_cache,
        index_k_cache,
        float(index_k_scale) if index_k_fp8 and index_k_scale is not None else 1.0,
        iq_mx,
        iq_sf,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        index_q.stride(0),
        index_k.stride(0),
        q_out.stride(0),
        iq_out.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.shape[1],
        num_actual_tokens,
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        num_index_heads,
        head_dim,
        half_rotary,
        eps,
        index_eps,
        INDEX_K_FP8=index_k_fp8,
        INDEX_K_MXFP6=index_k_mxfp6,
        INDEX_Q_MXFP8=index_q_mxfp8,
        FP8_MAX=float(torch.finfo(torch.float8_e4m3fn).max),
        INPUT_DTYPE=tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        HALF_BLOCK=half_block,
        HEAD_MAJOR=head_major,
        TOKENS_PER_PROGRAM=tokens_per_program,
        MX_PAGE_BYTES=layout["PAGE_BYTES"],
        MX_CODE_BYTES=layout["CODE_BYTES"],
        MX_CODE_REGION=layout["CODE_REGION"],
        MX_PAGE_TOKENS=layout["TOKENS"],
        n_tokens=n_tokens,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )
    return q_out, iq_out


def _fused_qk_rmsnorm_rope_cache_index_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_k_scale: float | None,
    num_actual_tokens: int,
    eps: float,
    index_eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    head_dim: int,
    q_out_dtype: torch.dtype | None = None,
    index_q_out_dtype: torch.dtype | None = None,
    index_q_mx: torch.Tensor | None = None,
    index_q_sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens = q.shape[0]
    return (
        q.new_empty(
            (n_tokens, num_q_heads * head_dim),
            dtype=q.dtype if q_out_dtype is None else q_out_dtype,
        ),
        q.new_empty(
            (n_tokens, num_index_heads, head_dim),
            dtype=q.dtype if index_q_out_dtype is None else index_q_out_dtype,
        ),
    )


# The caches and MXFP8 index-Q buffers are written in place and never read
# back in the graph, so they must be declared mutated for torch.compile.
direct_register_custom_op(
    op_name="maxwell_v1_flash_fused_qk_norm_rope_cache_index",
    op_func=fused_qk_rmsnorm_rope_cache_index,
    mutates_args=[
        "key_cache",
        "value_cache",
        "index_k_cache",
        "index_q_mx",
        "index_q_sf",
    ],
    fake_impl=_fused_qk_rmsnorm_rope_cache_index_fake,
)
