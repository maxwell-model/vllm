# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""The fused QKV + index projection for Maxwell-V1-Flash sparse layers."""

from typing import Any

import torch
from torch.nn.parameter import Parameter
from typing_extensions import TypeIs

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter

MAIN_SHARDS = ("q", "k", "v")
INDEX_SHARDS = ("index_q", "index_k")
FUSED_SHARDS = MAIN_SHARDS + INDEX_SHARDS


class MaxwellV1FlashQKVParallelLinear(QKVParallelLinear):
    """QKV projection fused with the indexer's index_q/index_k.

    One column-parallel GEMM emits ``[q | k | v | index_q | index_k]`` per rank.
    ``index_q`` has the KV head count and head size, so it shards (and
    replicates when ``tp_size > total_num_kv_heads``) exactly like K/V;
    ``index_k`` is a single head replicated to every rank.

    ``shards`` selects a subset, in that order, for projections whose weights
    do not share one format and so cannot share one GEMM.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        shards: tuple[str, ...] = FUSED_SHARDS,
    ) -> None:
        assert shards and all(s in FUSED_SHARDS for s in shards), shards
        assert total_num_index_heads == total_num_kv_heads, (
            "MaxwellV1FlashQKVParallelLinear requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.num_index_heads = self.num_kv_heads
        self.shards = shards

        widths = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
            "index_q": self.num_index_heads * self.index_head_size,
            "index_k": self.index_head_size,
        }
        self._shard_sizes = {s: widths[s] for s in shards}
        self._shard_offsets = {}
        offset = 0
        for s in shards:
            self._shard_offsets[s] = offset
            offset += widths[s]
        self.output_sizes = [widths[s] * tp_size for s in shards]

        # Bypass QKVParallelLinear.__init__, which assumes 3 groups.
        ColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def validate_shard_id(self, shard_id: Any) -> TypeIs[str | None]:
        if shard_id in self.shards or shard_id is None:
            return True
        raise ValueError(
            "Shard id for MaxwellV1FlashQKVParallelLinear must be one of "
            f"{self.shards}; got {shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        return self._shard_offsets.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return self._shard_sizes.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in self.shards
        _check_shard_dtype(self, param, loaded_weight, loaded_shard_id)

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        # num_heads == tp_size makes load_qkv_weight read shard 0 on every rank.
        num_heads = (
            self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in self.shards
        _check_shard_dtype(self, param, loaded_weight, loaded_shard_id)
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0  # replicated to every rank
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


def _check_shard_dtype(
    layer: MaxwellV1FlashQKVParallelLinear,
    param: Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
) -> None:
    # copy_ would silently cast a BF16 shard into an FP8 weight (or drop an
    # FP8 shard's scales into a BF16 one); a shard stored in another format
    # needs its own GEMM.
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    if param.dtype != loaded_weight.dtype and (
        param.dtype in fp8 or loaded_weight.dtype in fp8
    ):
        raise ValueError(
            f"{layer.prefix}: checkpoint shard {shard_id!r} is "
            f"{loaded_weight.dtype} but the fused weight is {param.dtype}; "
            "a fused projection needs every shard in one format."
        )
