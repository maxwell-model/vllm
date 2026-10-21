# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""vLLM mHC operators adapted to use ``1 / (sqrt(ms) + eps)`` gate scaling."""

import math

import torch

from vllm.model_executor.kernels.mhc.tilelang import _hc_prenorm_gemm_outputs
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    ENABLE_PDL,
    MhcPreBigFuseTileLangKernel,
)
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPreOp
from vllm.models.maxwell_v1_flash.nvidia.ops.mega_mhc import (
    MEGA_MHC_MIN_TOKENS,
    is_mega_mhc_supported,
    mhc_fused_post_pre_mega,
)
from vllm.platforms import current_platform
from vllm.tilelang_utils import T, tilelang_jit
from vllm.utils.torch_utils import direct_register_custom_op

# Thread predicates for the mix and collapse halves.
_MIX_ON_TRAILING_THREADS = current_platform.is_rocm()


def _is_mix_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() >= 64
    return T.get_thread_binding() < 32


def _is_collapse_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() < 64
    return T.get_thread_binding() >= 32


@tilelang_jit
def _mhc_pre_big_fuse_with_norm_eps_outside(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    pre_mix_in,
    pre_mix_out,
    aux_out,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    use_pre_mix_in: bool = False,
    save_pre_mix: bool = False,
    rms_numel: int = 0,
    split_mode: str = "fused",
    write_aux: bool = False,
):
    # Split modes require shifted mHC: input collapse uses a carried pre-mix.
    assert split_mode in ("fused", "stats", "input")
    assert split_mode == "fused" or save_pre_mix
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    pre_mix_in: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    aux_out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        if split_mode != "input":
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0

            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            # 1 / (sqrt(ms) + eps), written as r / (1 + eps * r) with
            # r = rsqrt(ms).
            rms[0] = T.rsqrt(rms[0] / rms_numel)
            rms[0] = rms[0] / (1.0 + rms_eps * rms[0])
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            T.copy(mixes, mixes_shared)

        if split_mode != "input" and _is_mix_thread():
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                if save_pre_mix:
                    pre_mix_out[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        elif split_mode != "stats" and _is_collapse_thread():
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                if use_pre_mix_in:
                    pre_mix_shared[j] = pre_mix_in[i, j]
                elif save_pre_mix:
                    pre_mix_shared[j] = T.if_then_else(j == 0, 1.0, 0.0)
                else:
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )

            # Pass 1: stash unnormalized weighted-sum output in shared memory
            # as bf16 (matches the rounding that RMSNorm would see) while
            # accumulating the per-position squared sum.
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                if write_aux:
                    # Aux consumers (draft models) take the plain stream mean
                    # of the same residual this collapse reads.
                    aux = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(aux)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                        if write_aux:
                            aux[i1_h] += xl[i_hc, i1_h]

                if write_aux:
                    for i1_h in T.Parallel(hidden_block):
                        aux_out[i, i0_h * hidden_block + i1_h] = T.bfloat16(
                            aux[i1_h] / hc_mult
                        )

                if save_pre_mix:
                    # Keep the BF16 boundary before the delayed input RMSNorm.
                    rounded = T.alloc_fragment(hidden_block, T.bfloat16)
                    T.copy(ol, rounded)
                    for i1_h in T.Parallel(hidden_block):
                        value = T.float32(rounded[i1_h])
                        sumsq_per_pos[i1_h] += value * value
                        output_shared[i0_h * hidden_block + i1_h] = rounded[i1_h]
                else:
                    for i1_h in T.Parallel(hidden_block):
                        sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                        output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


class _MhcPreBigFuseEpsOutsideKernel(MhcPreBigFuseTileLangKernel):
    """The shared big-fuse kernel owner, pointed at the eps-outside body.

    A subclass so its JIT cache is separate from the shared kernel's.
    """

    @staticmethod
    def kernel(compile_key: "MhcPreBigFuseTileLangKernel.CompileKey"):
        if compile_key.is_broadcast or not compile_key.use_norm_weight:
            raise NotImplementedError(
                "the eps-outside mHC kernel is only built with a norm weight"
            )
        return _mhc_pre_big_fuse_with_norm_eps_outside


EPS_OUTSIDE_BIG_FUSE_KERNEL = _MhcPreBigFuseEpsOutsideKernel()


def mhc_pre_eps_outside(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: retained for the shared MHC operator API; the active GEMM
            backend selects its split factor internally.
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16

    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
    )
    EPS_OUTSIDE_BIG_FUSE_KERNEL(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_eps_outside_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_fused_post_pre_eps_outside(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one MHC post block followed by the next MHC pre block.

    ``layer_input_cur`` is RMSNorm'd when ``norm_weight`` is given.
    ``n_splits`` and ``tile_n`` are kept for the shared MHC operator API.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)

    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        _MHC_POST_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)

    # Decided inside the custom op so the token count is not frozen by tracing.
    if (
        norm_weight is not None
        and residual.numel() // (hc_mult * hidden_size) >= MEGA_MHC_MIN_TOKENS
        and is_mega_mhc_supported(hidden_size, hc_mult)
    ):
        return mhc_fused_post_pre_mega(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    fused_config = mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)

    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if fused_config is not None:
        gemm_out_mul, gemm_out_sqrsum, residual_cur = _MHC_FUSED_TILELANG_KERNEL(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            hc_mult,
            hidden_size,
            hc_mult3,
        )
        EPS_OUTSIDE_BIG_FUSE_KERNEL(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    else:
        residual_cur = torch.empty_like(residual_flat)
        _MHC_POST_TILELANG_KERNEL(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
        )
        gemm_out_mul, gemm_out_sqrsum = _hc_prenorm_gemm_outputs(
            residual_cur.view(num_tokens, hc_mult * hidden_size),
            fn,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
        )
        EPS_OUTSIDE_BIG_FUSE_KERNEL(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_eps_outside_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def mhc_pre_fp32(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native mHC pre block with the epsilon outside the root."""
    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    outer_shape = residual.shape[:-2]
    streams = residual.reshape(-1, hc_mult, hidden_size).float()
    x = streams.flatten(1)
    rms = x.square().mean(dim=-1, keepdim=True).sqrt()
    mixes = (x @ fn.t()) / (rms + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    pre = pre + hc_pre_eps
    post = torch.sigmoid(
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post = post * hc_post_mult_value
    comb_logits = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * streams).sum(dim=1).to(torch.bfloat16)
    if norm_weight is not None:
        h = layer_input.float()
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = (h * norm_weight.float()).to(torch.bfloat16)
    return (
        post.view(*outer_shape, hc_mult, 1),
        comb.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_fused_post_pre_fp32(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One mHC post block then the next pre block, in eager fp32."""
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    post_in = post_layer_mix.reshape(*residual.shape[:-1], 1).float()
    residual_cur = (mixed + post_in * x.unsqueeze(-2).float()).to(residual.dtype)
    post, comb, layer_input = mhc_pre_fp32(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )
    return residual_cur, post, comb, layer_input


direct_register_custom_op(
    op_name="maxwell_v1_flash_mhc_pre",
    op_func=mhc_pre_eps_outside,
    mutates_args=[],
    fake_impl=_mhc_pre_eps_outside_fake,
)
direct_register_custom_op(
    op_name="maxwell_v1_flash_mhc_fused_post_pre",
    op_func=mhc_fused_post_pre_eps_outside,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_eps_outside_fake,
)


class MaxwellV1FlashMHCPreOp(MHCPreOp):
    """``MHCPreOp`` with this checkpoint's epsilon placement."""

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.maxwell_v1_flash_mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_pre_fp32(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )


class MaxwellV1FlashMHCFusedPostPreOp(MHCFusedPostPreOp):
    """``MHCFusedPostPreOp`` with this checkpoint's epsilon placement."""

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.maxwell_v1_flash_mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_fused_post_pre_fp32(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
