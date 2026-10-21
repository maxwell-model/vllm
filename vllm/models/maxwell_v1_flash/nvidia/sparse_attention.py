# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Block-sparse GQA with MBSA prefill, Triton fallback and split-K decode."""

import importlib
import importlib.util
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.models.maxwell_v1_flash import envs
from vllm.models.maxwell_v1_flash.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    maxwell_sparse_attn_decode,
    maxwell_sparse_attn_prefill,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

logger = init_logger(__name__)

#: Page counts the sparse attend kernel is instantiated for.
SUPPORTED_TOPK_BLOCKS = (4, 8, 16, 32)

#: Per-rank (query heads, KV heads) MBSA is built for.
MBSA_HEADS = (32, 4)

_MBSA = "vllm.third_party.mbsa"


def mbsa_serving() -> ModuleType:
    """MBSA's prefill module (``mbsa.serving``)."""
    return importlib.import_module(f"{_MBSA}.serving")


def mbsa_prefill_reason(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    topk_blocks: int,
    sparse_block_size: int,
    init_blocks: int,
    local_blocks: int,
    share_route: bool,
    score_sum: bool,
    index_format: str,
    scale: float,
) -> str | None:
    """Why this layer cannot prefill through MBSA prefill, or None.

    MBSA prefill picks pages the way the indexer does only for per-head
    max-reduced routes with 8 prefix and 8 local pages out of 32.
    """
    if output_dtype != torch.bfloat16:
        return "MBSA requires BF16 attention output"
    if importlib.util.find_spec(_MBSA) is None:
        return "this build does not ship MBSA"
    capability = current_platform.get_device_capability()
    if capability is None:
        return "no GPU capability is known"
    if capability.major == 9:
        return "MBSA's SM90 prefill is slower than the Triton path"
    return mbsa_serving().unsupported_reason(
        capability=(capability.major, capability.minor),
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        page_size=sparse_block_size,
        topk_pages=topk_blocks,
        prefix_pages=init_blocks,
        local_pages=local_blocks,
        kv_dtype=kv_dtype,
        index_format=index_format,
        softmax_scale=scale,
        page_score="sum" if score_sum else "max",
        shared_route=share_route,
    )


def mbsa_partial_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if envs.MBSA_PARTIALS_FP8 else torch.bfloat16


def mbsa_prefill_runner(device: torch.device) -> Any:
    """The device's shared MBSA prefill runner."""
    if device.index is None:
        device = torch.device(device.type, torch.accelerator.current_device_index())
    return mbsa_serving().runner(device, mbsa_partial_dtype())


class MaxwellV1FlashSparseBackend(AttentionBackend):
    """Block-sparse GQA backend for Maxwell-V1-Flash attention layers."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "MAXWELL_V1_FLASH_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MaxwellV1FlashSparseImpl"]:
        return MaxwellV1FlashSparseImpl

    @staticmethod
    def get_builder_cls() -> type["MaxwellV1FlashSparseMetadataBuilder"]:
        return MaxwellV1FlashSparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # One sparse block per KV page.
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def is_sparse(cls) -> bool:
        return True


@dataclass
class MaxwellV1FlashSparsePrefillMetadata:
    """Paged request metadata shared by MBSA and generic Triton prefill."""

    cu_seqlens_q: torch.Tensor  # [num_prefills + 1] int32, rebased to 0
    seq_lens: torch.Tensor  # [num_prefills] int32, total KV lengths
    context_lens: torch.Tensor  # [num_prefills] int32, cached tokens
    block_table: torch.Tensor
    max_query_len: int
    #: Host ``(query, KV)`` lengths per request, for MBSA prefill.
    request_lens: tuple[tuple[int, int], ...] = ()
    #: The step's staged MBSA prefill (or None), by index block table.
    mbsa_prefill_steps: dict[int, Any] = field(default_factory=dict)


@dataclass
class MaxwellV1FlashSparseDecodeMetadata:
    """Per-decode state (cudagraph-safe). ``decode_query_len`` is the uniform
    per-request query length (1, or 1 + num_speculative_tokens)."""

    seq_lens: torch.Tensor  # [num_decodes] int32
    block_table: torch.Tensor
    decode_query_len: int


@dataclass
class MaxwellV1FlashSparseMetadata(AttentionMetadata):
    """Sparse-attention metadata, split into prefill and decode sub-metadata."""

    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int  # total query tokens (decode-first batch)
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: MaxwellV1FlashSparsePrefillMetadata | None = None
    decode: MaxwellV1FlashSparseDecodeMetadata | None = None


class MaxwellV1FlashSparseMetadataBuilder(
    AttentionMetadataBuilder[MaxwellV1FlashSparseMetadata]
):
    # Full cudagraphs for uniform decode batches, incl. spec-decode verify
    # batches with >1 query token/request.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Raised to 1 + num_speculative_tokens by _init_reorder_batch_threshold when
    # spec decode is on; must match the indexer builder so the splits agree.
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
    ) -> MaxwellV1FlashSparseMetadata:
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

        prefill_metadata: MaxwellV1FlashSparsePrefillMetadata | None = None
        if num_prefills > 0:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens = (
                qsl_cpu[num_decodes + 1 : num_reqs + 1] - qsl_cpu[num_decodes:num_reqs]
            )
            request_lens = tuple(
                zip(query_lens.tolist(), seq_lens_cpu[num_decodes:num_reqs].tolist())
            )
            prefill_kv_lens = seq_lens[num_decodes:]
            prefill_metadata = MaxwellV1FlashSparsePrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=prefill_kv_lens,
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                request_lens=request_lens,
            )

        decode_metadata: MaxwellV1FlashSparseDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            assert decode_query_len > 0
            assert torch.all(
                (query_lens_cpu == decode_query_len) | (query_lens_cpu == 0)
            )
            assert num_decode_tokens == num_decodes * decode_query_len
            decode_metadata = MaxwellV1FlashSparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                decode_query_len=decode_query_len,
            )

        return MaxwellV1FlashSparseMetadata(
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


def _kv_planes(kv_cache: torch.Tensor, head_size: int) -> torch.Tensor:
    """View ``[B, Hkv, N, 2 * D]`` as the ``[B, 2, N, Hkv, D]`` decode reads."""
    blocks, heads, positions, packed = kv_cache.shape
    if packed != 2 * head_size:
        raise ValueError(
            f"expected K and V packed into {2 * head_size} columns, got {packed}"
        )
    s_blk, s_h, s_pos, s_d = kv_cache.stride()
    if s_d != 1:
        raise ValueError(f"expected a packed head dim, got stride {s_d}")
    return kv_cache.as_strided(
        (blocks, 2, positions, heads, head_size),
        (s_blk, head_size, s_pos, s_h, 1),
    )


class MaxwellV1FlashSparseImpl(AttentionImplBase[MaxwellV1FlashSparseMetadata]):
    """Sparse prefill (MBSA where supported) plus Triton sparse decode.

    The layer has already written K/V and run the indexer, which leaves the
    selected pages in ``layer.topk_indices_buffer``; the attend reads them from
    there. A prefill given a ``mbsa_prefill_step`` instead runs MBSA prefill, which
    selects its pages from the index cache itself.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv = is_quantized_kv_cache(kv_cache_dtype)
        if "e5m2" in kv_cache_dtype:
            self.kv_cache_fp8_dtype = torch.float8_e5m2
        else:
            self.kv_cache_fp8_dtype = current_platform.fp8_dtype()
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        #: MBSA ``serving.Limits`` when MBSA prefill may run, else None.
        self.mbsa_prefill_limits: Any = None
        self.mbsa_prefill_index_format = ""

    def configure_mbsa_prefill(
        self,
        *,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        index_format: str,
        init_blocks: int,
        local_blocks: int,
        share_route: bool,
        score_sum: bool,
        max_num_tokens: int,
        max_model_len: int,
        max_num_seqs: int,
    ) -> str | None:
        """Decide once whether this layer may prefill through MBSA prefill.

        Returns:
            None when it may, else why not; a batch MBSA prefill cannot take
            uses the generic Triton attend per step.

        """
        reason = mbsa_prefill_reason(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            kv_dtype=kv_dtype,
            output_dtype=output_dtype,
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.block_size,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            share_route=share_route,
            score_sum=score_sum,
            index_format=index_format,
            scale=self.scale,
        )
        if reason is None:
            self.mbsa_prefill_index_format = index_format
            self.mbsa_prefill_limits = mbsa_serving().Limits(
                max_tokens=max_num_tokens,
                max_kv_tokens=max_model_len,
                max_requests=max_num_seqs,
                max_total_kv_tokens=max(
                    min(max_num_seqs, max_num_tokens) * max_model_len, max_num_tokens
                ),
            )
        else:
            logger.info_once(
                "Maxwell-V1-Flash uses generic Triton prefill: %s.", reason
            )
        return reason

    def mbsa_prefill_step(self, layer: AttentionLayer) -> Any:
        """This step's staged MBSA prefill, or None to use generic Triton prefill.

        Decided once per step and shared by every layer, so the indexer and
        the attend agree on who selects the prefill pages.
        """
        if self.mbsa_prefill_limits is None:
            return None
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MaxwellV1FlashSparseMetadata)
        p = main_md.prefill
        if p is None or main_md.num_prefills == 0:
            return None
        index_prefix = layer.indexer.index_cache.prefix  # type: ignore[attr-defined]
        index_md = attn_metadata[index_prefix]
        index_table = index_md.prefill.block_table  # type: ignore[attr-defined]
        key = index_table.data_ptr()
        if key not in p.mbsa_prefill_steps:
            p.mbsa_prefill_steps[key] = self._stage_mbsa_prefill(main_md, index_table)
        return p.mbsa_prefill_steps[key]

    def _stage_mbsa_prefill(
        self, main_md: MaxwellV1FlashSparseMetadata, index_table: torch.Tensor
    ) -> Any:
        p = main_md.prefill
        assert p is not None
        reason = self.mbsa_prefill_limits.batch_reason(p.request_lens)
        if (
            reason is None
            and main_md.num_decode_tokens + sum(q_len for q_len, _ in p.request_lens)
            > main_md.num_actual_tokens
        ):
            reason = "the prefill lengths exceed the batch's tokens"
        elif reason is None and not p.block_table.is_contiguous():
            reason = "the block table is not contiguous"
        if reason is not None:
            logger.info_once(
                "Maxwell-V1-Flash uses generic Triton prefill: %s.", reason
            )
            return None
        logger.info_once("Maxwell-V1-Flash runs MBSA prefill.")
        return mbsa_prefill_runner(p.block_table.device).stage(
            p.request_lens,
            p.block_table,
            index_table,
        )

    def reserve_profile_workspace(self, num_tokens: int, device: torch.device) -> None:
        """Reserve the prepared prefill scratch during memory profiling."""
        if self.mbsa_prefill_limits is not None:
            size = self.mbsa_prefill_limits.workspace_bytes(
                partial_dtype=mbsa_partial_dtype(),
                index_format=self.mbsa_prefill_index_format,
            )
            torch.empty(size, dtype=torch.uint8, device=device)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
        index_query: torch.Tensor | None = None,
        mbsa_prefill_step: Any = None,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MaxwellV1FlashSparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        # Selected pages: decode at [:nd], prefill at [nd:num_tokens]; the
        # kernels want the head dimension outermost.
        topk_buffer = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk_buffer is not None
        topk = topk_buffer[:num_tokens].transpose(0, 1)
        if self.use_fp8_kv:
            kv_cache = kv_cache.view(self.kv_cache_fp8_dtype)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            maxwell_sparse_attn_decode(
                q[:nd],
                _kv_planes(kv_cache, hd),
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None
            if mbsa_prefill_step is not None:
                indexer = layer.indexer  # type: ignore[attr-defined]
                if indexer.index_q_mxfp8 is not None:
                    q_mx, q_sf = indexer.index_q_mxfp8
                    index_q, index_q_scales = q_mx[nd:, : self.num_kv_heads], q_sf[nd:]
                else:
                    assert index_query is not None
                    index_q = index_query[:num_tokens].view(-1, self.num_kv_heads, hd)
                    index_q, index_q_scales = index_q[nd:], None
                mbsa_prefill(
                    mbsa_prefill_step,
                    q[nd:],
                    kv_cache,
                    out[nd:],
                    index_q=index_q,
                    index_k_cache=indexer.index_cache.kv_cache,
                    index_q_scales=index_q_scales,
                )
            else:
                maxwell_sparse_attn_prefill(
                    q[nd:],
                    kv_cache,
                    topk[:, nd:num_tokens, :],
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.max_query_len,
                    self.scale,
                    out[nd:],
                )
        return output


def mbsa_prefill(
    step: Any,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    index_q: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_q_scales: torch.Tensor | None = None,
) -> None:
    """Prefill through MBSA prefill, which selects its own pages.

    Args:
        step: The step's ``mbsa.serving.PrefillStep``.
        q: ``[>=T, Hq, D]`` prefill queries; the step's ``T`` rows are read.
        kv_cache: The ``[B, Hkv, N, 2 * D]`` BF16 or E4M3 cache.
        out: ``[>=T, Hq, D]`` output.
        index_q: ``[>=T, Hkv, D]`` index queries (or MXFP8 rows).
        index_k_cache: The ``[B, N, C]`` index cache.
        index_q_scales: ``[>=T, >=Hkv]`` MXFP8 scales for an MXFP6 cache.

    """
    rows = step.total_q
    key_cache, value_cache = kv_cache.split(q.shape[-1], dim=-1)
    step.run(
        q[:rows],
        key_cache,
        value_cache,
        index_q=index_q[:rows],
        index_k_cache=index_k_cache,
        out=out[:rows],
        index_q_scales=None if index_q_scales is None else index_q_scales[:rows],
    )


class MaxwellV1FlashMbsaPrefillKernel(
    VllmJitKernel["MaxwellV1FlashMbsaPrefillKernel.CompileKey"]
):
    """Compiles every kernel variant of MBSA prefill.

    MBSA lists one step geometry per variant within the serving limits; each
    runs once on dummy caches.
    """

    @dataclass(frozen=True)
    class CompileKey:
        index_format: str
        partial_fp8: bool
        request_lens: tuple[tuple[int, int], ...]
        kv_dtype: torch.dtype
        q_dtype: torch.dtype

    def get_warmup_keys(
        self,
        *,
        index_format: str,
        limits: Any,
        kv_dtype: torch.dtype,
        q_dtype: torch.dtype,
    ) -> list[CompileKey]:
        geometries = mbsa_serving().warmup_request_sets(
            limits, index_format=index_format
        )
        return [
            self.CompileKey(
                index_format, envs.MBSA_PARTIALS_FP8, lens, kv_dtype, q_dtype
            )
            for lens in geometries
        ]

    def compile(self, compile_key: CompileKey) -> None:
        device = torch.accelerator.current_accelerator()
        assert device is not None
        mbsa_serving().warmup(
            mbsa_prefill_runner(device),
            compile_key.request_lens,
            index_format=compile_key.index_format,
            kv_dtype=compile_key.kv_dtype,
            q_dtype=compile_key.q_dtype,
        )
        self._compiled_cache[compile_key] = True


MBSA_PREFILL_KERNEL = MaxwellV1FlashMbsaPrefillKernel()
