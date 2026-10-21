# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Maxwell-V1-Flash model package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .nvidia.model import MaxwellV1FlashForCausalLM
    from .nvidia.mtp import MaxwellV1FlashMTP


def __getattr__(name: str) -> Any:
    if name in ("MaxwellV1FlashForCausalLM", "MaxwellV1FlashMTP"):
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            raise NotImplementedError(
                "Maxwell-V1-Flash currently supports CUDA only; its sparse "
                "attention is built on SM100 kernels."
            )
        if name == "MaxwellV1FlashMTP":
            from .nvidia.mtp import MaxwellV1FlashMTP

            return MaxwellV1FlashMTP
        from .nvidia.model import MaxwellV1FlashForCausalLM

        return MaxwellV1FlashForCausalLM
    raise AttributeError(name)


__all__ = ["MaxwellV1FlashForCausalLM", "MaxwellV1FlashMTP"]
