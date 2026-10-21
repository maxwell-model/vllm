# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Maxwell-V1-Flash runtime switches, read once when the package loads.

``MAXWELL_MBSA_INDEX`` (bf16/fp16/fp8/mxfp6, default fp8)
    Storage format of the index-K cache. An explicit
    ``--attention-config '{"indexer_kv_dtype": ...}'`` still takes precedence.
``MAXWELL_MBSA_PARTIALS_FP8`` (0/1, default 1)
    1 stores Dynamic partial outputs in FP8 on the architecture's prefill path.
"""

import os

INDEX_FORMATS = ("bf16", "fp16", "fp8", "mxfp6")


def _flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, default)
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


MBSA_INDEX: str = _choice("MAXWELL_MBSA_INDEX", "fp8", INDEX_FORMATS)

MBSA_PARTIALS_FP8: bool = _flag("MAXWELL_MBSA_PARTIALS_FP8", "1")
