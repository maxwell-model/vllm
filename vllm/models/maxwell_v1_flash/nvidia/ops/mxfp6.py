# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Packed MXFP6 index-K cache for Maxwell-V1-Flash (``MAXWELL_MBSA_INDEX=mxfp6``).

Index K is Hadamard-rotated and stored as MXFP6-E2M3 codes with one UE8M0
scale per 32 elements; index Q is rotated the same way and quantized to
MXFP8-E4M3. The format, writers and scorers live in MBSA, vendored at
``vllm.third_party.mbsa``.
"""

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

_MBSA = "vllm.third_party.mbsa"

if HAS_TRITON and importlib.util.find_spec(_MBSA) is not None:
    from vllm.third_party.mbsa.mxfp6.write import (
        mxfp6_encode_store,
        mxfp8_q_encode_store,
        mxfp8_q_pad_store,
    )
else:
    # Only reached under constexpr MXFP6 flags, which page_layout() rejects
    # without MBSA.
    mxfp6_encode_store = mxfp8_q_encode_store = mxfp8_q_pad_store = None


def _mbsa(name: str) -> ModuleType:
    try:
        return importlib.import_module(f"{_MBSA}.{name}")
    except ImportError as exc:
        raise ImportError(
            f"indexer_kv_dtype=mxfp6 needs MBSA's {name}, vendored at "
            f"{_MBSA} by the CUDA build; this installation does not include it."
        ) from exc


def index_formats() -> ModuleType:
    """MBSA's MXFP6 codec and page layout (``mbsa.index_formats``).

    Raises:
        ImportError: if this vLLM build does not ship MBSA.

    """
    return _mbsa("index_formats")


def kernels() -> ModuleType:
    """MBSA's MXFP6 page scorers and their limits (``mbsa.mxfp6``).

    Raises:
        ImportError: if this vLLM build does not ship MBSA.

    """
    return _mbsa("mxfp6")


def page_layout() -> dict[str, int]:
    """The packed page layout as the MXFP6 encoders' constexpr arguments."""
    return _mbsa("mxfp6.write").page_layout()


def shared_schedule(seq_lens: torch.Tensor, next_n: int) -> torch.Tensor:
    """The decode scorer's schedule, computed once per forward pass.

    Keyed on the forward context rather than ``seq_lens``, which is a
    persistent buffer reused with new contents every pass.
    """
    owner = get_forward_context() if is_forward_context_available() else None
    return kernels().shared_schedule(seq_lens, next_n, owner)


def _arch() -> str:
    cap = current_platform.get_device_capability()
    if cap is None or cap.major != 10:
        raise RuntimeError(
            f"the MXFP6 index scorers need an SM100-family GPU, got {cap}"
        )
    return f"{cap.major}{cap.minor}a"


class MaxwellV1FlashMxfp6IndexKernel(
    VllmJitKernel["MaxwellV1FlashMxfp6IndexKernel.CompileKey"]
):
    """Loads both MXFP6 scorer extensions before serving.

    MBSA JIT-compiles a missing prebuilt extension, which must not happen on
    the first request.
    """

    @dataclass(frozen=True)
    class CompileKey:
        arch: str

    def get_warmup_keys(self) -> list[CompileKey]:
        return [self.CompileKey(_arch())]

    def compile(self, compile_key: CompileKey) -> None:
        kernels().compile_extensions()
        self._compiled_cache[compile_key] = True


MXFP6_INDEX_KERNEL = MaxwellV1FlashMxfp6IndexKernel()
