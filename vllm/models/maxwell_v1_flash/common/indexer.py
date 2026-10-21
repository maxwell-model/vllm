# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Maxwell-V1-Flash block-sparse indexer: side cache, metadata, and impl.

The indexer scores KV pages with its own index heads and selects the top-k
pages plus the pinned init/local windows for the main block-sparse attention.
Its key-only side cache stores BF16, FP8 E4M3, or packed MXFP6.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.maxwell_v1_flash.common.ops.index_topk import (
    maxwell_index_decode,
    maxwell_index_score,
    maxwell_index_topk,
    maxwell_index_topk_from_scores,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

logger = init_logger(__name__)

#: The Triton scorers fold log2(e) into the scale; the MXFP6 ones do not.
_LOG2E = 1.4426950409


#: ``block_sparse_gqa_group_reduction`` -> (score_sum, share_route).
_PAGE_REDUCTIONS = {
    "max": (False, True),
    "sum": (True, True),
    "raw_sum": (True, True),
    "per_group_max": (False, False),
}


def page_reduction(group_reduction: str) -> tuple[bool, bool]:
    """Return ``(score_sum, share_route)`` for a checkpoint's routing topology.

    Raises:
        ValueError: For an unrecognized value, which would otherwise silently
            degrade accuracy.

    """
    try:
        return _PAGE_REDUCTIONS[group_reduction]
    except KeyError:
        raise ValueError(
            "block_sparse_gqa_group_reduction must be one of "
            f"{sorted(_PAGE_REDUCTIONS)}, got {group_reduction!r}"
        ) from None


@dataclass
class MaxwellV1FlashIndexerPrefillMetadata:
    """Per-prefill index-scoring state."""

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32, total KV lengths
    context_lens: torch.Tensor  # [num_prefills] int32, cached tokens
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class MaxwellV1FlashIndexerDecodeMetadata:
    """Per-decode state (cudagraph-safe). ``decode_query_len`` is the uniform
    per-request query length (1, or 1 + num_speculative_tokens)."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int


@dataclass
class MaxwellV1FlashIndexerMetadata(AttentionMetadata):
    """Indexer metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int  # total query tokens (decode-first batch)
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: MaxwellV1FlashIndexerPrefillMetadata | None = None
    decode: MaxwellV1FlashIndexerDecodeMetadata | None = None


class MaxwellV1FlashIndexerMetadataBuilder(
    AttentionMetadataBuilder[MaxwellV1FlashIndexerMetadata]
):
    """Metadata for the key-only, single-head index cache."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Must match the main builder so the decode/prefill splits agree.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        # Stable context-length buffer for decode cudagraph replays.
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MaxwellV1FlashIndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: MaxwellV1FlashIndexerPrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = MaxwellV1FlashIndexerPrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: MaxwellV1FlashIndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            assert num_decode_tokens == num_decodes * decode_query_len
            decode_metadata = MaxwellV1FlashIndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return MaxwellV1FlashIndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class MaxwellV1FlashIndexerBackend(AttentionBackend):
    """Index side-cache backend (key-only)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "MAXWELL_V1_FLASH_INDEXER"

    @staticmethod
    def get_impl_cls() -> type:
        raise NotImplementedError("the indexer runs through MaxwellV1FlashIndexer")

    @staticmethod
    def get_builder_cls() -> type[MaxwellV1FlashIndexerMetadataBuilder]:
        return MaxwellV1FlashIndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class MaxwellV1FlashIndexerCache(nn.Module, AttentionLayerBase):
    """Side KV cache for the indexer's per-token index keys (key-only).

    Registers itself in the static forward context so the KV-cache manager
    allocates it.
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
    ) -> None:
        super().__init__()
        storage_width = head_dim
        if indexer_kv_dtype in ("fp8", "fp8_e4m3"):
            cache_dtype = torch.float8_e4m3fn
        elif indexer_kv_dtype == "bf16":
            cache_dtype = torch.bfloat16
        elif indexer_kv_dtype == "fp16":
            cache_dtype = torch.float16
        elif indexer_kv_dtype == "mxfp6":
            from vllm.models.maxwell_v1_flash.nvidia.ops.mxfp6 import index_formats

            if head_dim != 128:
                raise ValueError(
                    f"the MXFP6 index cache packs 128-dim keys, got {head_dim}"
                )
            cache_dtype = torch.uint8
            storage_width = index_formats().BYTES_PER_TOKEN
        else:
            raise NotImplementedError(
                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "
                "index cache (only 'bf16', 'fp16', 'fp8'/'fp8_e4m3' or 'mxfp6')."
            )
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.storage_width = storage_width
        self.indexer_kv_dtype = indexer_kv_dtype
        self.dtype = cache_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector/token (not 2x for K+V).
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.storage_width,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MaxwellV1FlashIndexerBackend


class MaxwellV1FlashIndexer(nn.Module):
    """Page selection for one block-sparse attention layer.

    Owns the index-key side cache and writes the selected pages into the
    layer's shared top-k buffer.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        num_index_heads: int,
        index_head_dim: int,
        topk_blocks: int,
        sparse_block_size: int,
        init_blocks: int,
        local_blocks: int,
        group_reduction: str,
        prefix: str,
        rms_norm_eps: float,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType = "bf16",
        topk_indices_buffer: torch.Tensor,
        index_q_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if num_index_heads != num_kv_heads:
            raise ValueError(
                "the indexer selects one page route per KV head, so it needs "
                f"one index head per KV head; got {num_index_heads} index "
                f"heads and {num_kv_heads} KV heads."
            )
        if sparse_block_size != 128:
            raise ValueError(
                f"the sparse page size must be 128, got {sparse_block_size}"
            )
        self.num_kv_heads = num_kv_heads
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        # The checkpoint's page ranking is trained on index-head-dim scaled
        # logits.
        self.sm_scale = 1.0 / math.sqrt(index_head_dim)
        self.score_sum, self.share_route = page_reduction(group_reduction)
        self.topk_indices_buffer = topk_indices_buffer
        self.q_layernorm = RMSNorm(index_head_dim, eps=rms_norm_eps)
        self.k_layernorm = RMSNorm(index_head_dim, eps=rms_norm_eps)
        self.index_cache = MaxwellV1FlashIndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            indexer_kv_dtype=indexer_kv_dtype,
        )
        if self.index_mxfp6 and index_q_mxfp8 is None:
            raise ValueError("the MXFP6 index cache needs the MXFP8 index-Q buffers")
        #: ``([tokens, 8, 128] uint8, [tokens, 8] int32)`` index Q in MXFP8,
        #: written by the fused QK-norm-RoPE launch and shared by every layer.
        self.index_q_mxfp8 = index_q_mxfp8 if self.index_mxfp6 else None

    @property
    def index_k_fp8(self) -> bool:
        """Whether the index cache stores K in E4M3."""
        return self.index_cache.dtype is torch.float8_e4m3fn

    @property
    def index_mxfp6(self) -> bool:
        """Whether the index cache stores K as packed MXFP6."""
        return self.index_cache.dtype is torch.uint8

    def _reduce_route(self, score: torch.Tensor) -> torch.Tensor:
        """Collapse the per-head scores to the one row a shared route selects.

        The result is contiguous for the top-k kernel's strides.
        """
        if not self.share_route:
            return score
        return (
            score.sum(dim=0, keepdim=True)
            if self.score_sum
            else score.amax(dim=0, keepdim=True)
        )

    def forward(self, index_query: torch.Tensor, *, skip_prefill: bool = False) -> None:
        """Select pages for this step into the shared top-k buffer.

        Args:
            index_query: ``[tokens, heads * head_dim]`` index queries.
            skip_prefill: Select decode pages only, because the attention
                routes this step's prefill tokens itself (MBSA prefill).

        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return  # profiling run; caches unbound
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MaxwellV1FlashIndexerMetadata)
        if self.index_q_mxfp8 is not None:
            self._forward_mxfp6(index_md, *self.index_q_mxfp8, skip_prefill)
            return
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache
        # The buffer is token-major; the kernels want it head-major.
        buf_htk = self.topk_indices_buffer.transpose(0, 1)

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            maxwell_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.sm_scale,
                d.decode_query_len,
                share_route=self.share_route,
                score_sum=self.score_sum,
                out=buf_htk,
            )
        if index_md.num_prefills > 0 and not skip_prefill:
            p = index_md.prefill
            assert p is not None
            score = maxwell_index_score(
                iq[nd:],
                kv,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                p.max_seq_len,
                self.num_kv_heads,
                self.sm_scale,
            )
            self._select_prefill(score, p, buf_htk[:, nd:num_tokens, :])

    def _select_prefill(
        self,
        score: torch.Tensor,
        p: MaxwellV1FlashIndexerPrefillMetadata,
        dest: torch.Tensor,
    ) -> None:
        """Top-k prefill pages from per-head scores into ``dest``."""
        selected = maxwell_index_topk(
            self._reduce_route(score),
            p.cu_seqlens_q,
            p.context_lens,
            p.max_query_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            out=None if self.share_route else dest,
        )
        if self.share_route:
            # The CSR builder needs a materialized, not stride-zero, broadcast.
            dest.copy_(selected.expand(self.num_kv_heads, -1, -1))

    def _forward_mxfp6(
        self,
        index_md: MaxwellV1FlashIndexerMetadata,
        q_mx: torch.Tensor,
        q_sf: torch.Tensor,
        skip_prefill: bool = False,
    ) -> None:
        """Score the packed cache with the MXFP6 kernels, then select as usual.

        Reads the fused launch's rotated MXFP8 index Q, not the bf16 one.
        """
        from vllm.models.maxwell_v1_flash.nvidia.ops import mxfp6

        kernels = mxfp6.kernels()

        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        cache = self.index_cache.kv_cache.view(-1)
        buf_htk = self.topk_indices_buffer.transpose(0, 1)
        sm_scale = self.sm_scale * _LOG2E

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            scores = kernels.decode_score(
                q_mx[:nd],
                q_sf[:nd],
                cache,
                d.block_table,
                d.seq_lens,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                score_sum=self.score_sum,
                sm_scale=sm_scale,
                next_n=d.decode_query_len,
                schedule=mxfp6.shared_schedule(d.seq_lens, d.decode_query_len),
            )
            maxwell_index_topk_from_scores(
                scores,
                d.seq_lens,
                self.topk_blocks,
                self.num_kv_heads,
                -(-d.max_seq_len // self.block_size),
                d.decode_query_len,
                share_route=self.share_route,
                score_sum=self.score_sum,
                out=buf_htk,
            )
        if index_md.num_prefills > 0 and not skip_prefill:
            p = index_md.prefill
            assert p is not None
            score = kernels.prefill_score(
                q_mx[nd:num_tokens],
                q_sf[nd:num_tokens],
                cache,
                p.block_table,
                p.cu_seqlens_q,
                p.context_lens,
                max_seq_len=p.max_seq_len,
                score_sum=self.score_sum,
                sm_scale=sm_scale,
            )
            self._select_prefill(score, p, buf_htk[:, nd:num_tokens, :])
