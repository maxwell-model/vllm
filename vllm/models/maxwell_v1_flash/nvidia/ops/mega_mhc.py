# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""DeepGEMM Mega mHC: one kernel for mHC post + next mHC pre + RMSNorm.

This checkpoint normalizes the HC input as ``1 / (RMS(x) + eps)``, which
DeepGEMM's ``eps_norm_out`` selects. Only the normal (non-shifted) mode is
used, as a drop-in for the TileLang fused post+pre block.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported, mega_mhc

logger = init_logger(__name__)

#: Below this many tokens a step keeps the TileLang fused kernel.
MEGA_MHC_MIN_TOKENS = 2048


@functools.cache
def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    """Whether Mega mHC can run this geometry on this device.

    The kernel must also take ``eps_norm_out``, or its output is silently wrong.
    """
    if not (
        is_deep_gemm_supported()
        and current_platform.is_device_capability_family(100)
        and hidden_size > 0
        and hidden_size % 1024 == 0
        and hc_mult == 4
    ):
        return False
    impl = getattr(_import_deep_gemm(), "mega_mhc", None)
    if impl is None:
        return False
    if "eps_norm_out" not in (impl.__doc__ or ""):
        logger.warning("DeepGEMM Mega mHC lacks eps_norm_out; keeping TileLang mHC")
        return False
    logger.info(
        "Mega mHC enabled for steps of >= %d tokens (hidden_size=%d, hc_mult=%d)",
        MEGA_MHC_MIN_TOKENS,
        hidden_size,
        hc_mult,
    )
    return True


def mhc_fused_post_pre_mega(
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
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The TileLang fused post+pre contract, with ``eps_norm_out`` set.

    Returns:
        residual_cur, post_mix_cur, comb_mix_cur and the RMSNorm'd
        layer_input_cur.

    """
    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    outer_shape = residual.shape[:-2]

    post_mix_in = post_layer_mix.view(num_tokens, hc_mult, 1)
    new_residual = torch.empty_like(residual_flat)
    new_post_mix = torch.empty_like(post_mix_in)
    new_comb_res_mix = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    y_bf16 = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    mega_mhc(
        x=x.view(num_tokens, hidden_size),
        residual=residual_flat,
        shifted_prev_mix=None,
        post_mix=post_mix_in,
        comb_res_mix=comb_res_mix.view(num_tokens, hc_mult, hc_mult),
        fn=fn,
        mix_scales=hc_scale,
        mix_bases=hc_base,
        hc_mult=hc_mult,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_scale=hc_post_mult_value,
        sinkhorn_eps=hc_sinkhorn_eps,
        num_sinkhorn_iters=sinkhorn_repeat,
        rmsnorm_weight=norm_weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=None,
        new_post_mix=new_post_mix,
        new_comb_res_mix=new_comb_res_mix,
        y_bf16=y_bf16,
        eps_norm_out=True,
    )
    return (
        new_residual.view(*outer_shape, hc_mult, hidden_size),
        new_post_mix.view(*outer_shape, hc_mult, 1),
        new_comb_res_mix.view(*outer_shape, hc_mult, hc_mult),
        y_bf16.view(*outer_shape, hidden_size),
    )
