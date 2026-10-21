# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Inference-only Maxwell-V1-Flash MoE model.

Dense checkpoints and ``block_sparse_gqa`` checkpoints share this trunk. A
checkpoint without ``attention_variant`` attends densely on every layer; a
``block_sparse_gqa`` checkpoint adds an index branch that selects a causal
top-k of KV pages per query, and ``block_sparse_gqa_layer_freq`` marks which
layers were trained that way -- the rest stay dense.

Multi-stream hyper-connections (mHC) replace the plain residual when
``enable_hyper_connections`` is set. This checkpoint scales the mHC gates by
``1 / (sqrt(ms) + eps)`` rather than ``rsqrt(ms + eps)``, so the pre and fused
post+pre blocks come from this package's ``ops.mhc``.
"""

from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers import PreTrainedConfig

import vllm.envs as envs_vllm
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.attention import IndexerKVDType
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCPostOp,
    hc_expand,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.maxwell_v1_flash import envs
from vllm.models.maxwell_v1_flash.common.indexer import (
    MaxwellV1FlashIndexer,
    page_reduction,
)
from vllm.models.maxwell_v1_flash.common.linear import (
    FUSED_SHARDS,
    INDEX_SHARDS,
    MAIN_SHARDS,
    MaxwellV1FlashQKVParallelLinear,
)
from vllm.models.maxwell_v1_flash.nvidia.ops.fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope,
)
from vllm.models.maxwell_v1_flash.nvidia.ops.mhc import (
    EPS_OUTSIDE_BIG_FUSE_KERNEL,
    MaxwellV1FlashMHCFusedPostPreOp,
    MaxwellV1FlashMHCPreOp,
)
from vllm.models.maxwell_v1_flash.nvidia.ops.split_bf16_gemm import (
    split_router_weight,
)
from vllm.models.maxwell_v1_flash.nvidia.sparse_attention import (
    MBSA_PREFILL_KERNEL,
    SUPPORTED_TOPK_BLOCKS,
    MaxwellV1FlashSparseBackend,
    MaxwellV1FlashSparseImpl,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

#: The only ``attention_variant`` this implementation serves.
VARIANT_BLOCK_SPARSE_GQA = "block_sparse_gqa"

#: Default index side-cache dtype, from ``MAXWELL_MBSA_INDEX``. The index branch
#: only ranks pages, so it does not need the main cache's precision.
_INDEX_FORMATS: dict[str, IndexerKVDType] = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp8": "fp8",
    "mxfp6": "mxfp6",
}
DEFAULT_INDEXER_KV_DTYPE: IndexerKVDType = _INDEX_FORMATS[envs.MBSA_INDEX]

#: Scale the E4M3 index cache is written through. E4M3's relative precision is
#: range-independent and the normed index K stays far below the clamp, so a unit
#: scale needs no calibration.
INDEX_K_FP8_SCALE = 1.0

#: The mHC operators take three epsilons and a post multiplier; this
#: checkpoint uses one epsilon for all three and the standard multiplier.
_MHC_EPS = 1e-6
_MHC_POST_MULT = 2.0


def _head_dim(config: PreTrainedConfig) -> int:
    for attr in ("attention_head_size", "head_dim"):
        value = getattr(config, attr, None)
        if value:
            return int(value)
    return config.hidden_size // config.num_attention_heads


def _attention_variant(config: PreTrainedConfig) -> str | None:
    variant = getattr(config, "attention_variant", None)
    return str(variant) if variant else None


def _is_sparse_layer(config: PreTrainedConfig, layer_id: int) -> bool:
    """Whether this layer was trained with block-sparse attention.

    ``block_sparse_gqa_layer_freq`` marks the sparse layers; a zero means that
    layer attends densely. Absent, every layer is sparse.
    """
    if _attention_variant(config) != VARIANT_BLOCK_SPARSE_GQA:
        return False
    freq = getattr(config, "block_sparse_gqa_layer_freq", None)
    if freq is None or layer_id >= len(freq):
        return True
    return bool(freq[layer_id])


def indexer_kv_dtype(vllm_config: VllmConfig) -> IndexerKVDType:
    """The index side-cache dtype: ``--attention-config``, else the switch."""
    return vllm_config.attention_config.resolve_indexer_kv_dtype(
        DEFAULT_INDEXER_KV_DTYPE
    )


def _is_moe_layer(config: PreTrainedConfig, layer_id: int) -> bool:
    if config.num_experts == 0:
        return False
    return layer_id not in (getattr(config, "mlp_only_layers", None) or [])


def _topk_blocks(config: PreTrainedConfig) -> int:
    """Pages selected per query, pinned windows included."""
    topk = int(config.block_sparse_gqa_topk_blocks)
    return int(getattr(config, "block_sparse_gqa_total_support_blocks", topk) or topk)


def tp_head_counts(config: PreTrainedConfig, tp_size: int) -> tuple[int, int]:
    """Per-rank ``(query heads, KV heads)`` for a tensor-parallel size.

    KV heads are replicated once TP exceeds the KV head count. The index branch
    follows the same split.

    Raises:
        ValueError: if the TP size divides neither head count evenly.

    """
    total_heads = int(config.num_attention_heads)
    total_kv_heads = int(config.num_key_value_heads)
    if total_heads % tp_size:
        raise ValueError(
            f"--tensor-parallel-size {tp_size} must divide the "
            f"{total_heads} attention heads."
        )
    if total_kv_heads >= tp_size:
        if total_kv_heads % tp_size:
            raise ValueError(
                f"--tensor-parallel-size {tp_size} must divide the "
                f"{total_kv_heads} KV heads."
            )
    elif tp_size % total_kv_heads:
        raise ValueError(
            f"--tensor-parallel-size {tp_size} must be a multiple of the "
            f"{total_kv_heads} KV heads to replicate them."
        )
    return total_heads // tp_size, max(1, total_kv_heads // tp_size)


def _fused_norm_rope_supported(rotary_emb: nn.Module, head_dim: int) -> bool:
    """Whether the fused norm + RoPE kernel matches this RoPE.

    It rotates the full head with NeoX pairing, so a partial or GPT-J style
    rotation has to take the separate ops instead.
    """
    return bool(
        current_platform.is_cuda()
        and getattr(rotary_emb, "is_neox_style", False)
        and rotary_emb.rotary_dim == head_dim
    )


def _linear_quant_method_type(
    quant_config: QuantizationConfig | None, prefix: str
) -> type:
    if quant_config is None:
        return UnquantizedLinearMethod
    # LinearBase resolves the method without allocating weights.
    probe = LinearBase(1, 1, quant_config=quant_config, prefix=prefix, disable_tp=True)
    return type(probe.quant_method)


def _fuse_index_projection(
    quant_config: QuantizationConfig | None, prefix: str
) -> bool:
    """Whether index_q/index_k can join q/k/v in one GEMM.

    One GEMM holds one weight format. A checkpoint may quantize the main
    projections but keep the indexer's in BF16, so the index projections fuse
    only when they resolve to the main projection's quantization method; a
    BF16 indexer otherwise runs as its own GEMM, and any other mix is rejected.
    """
    fused = _linear_quant_method_type(quant_config, f"{prefix}.qkv_proj")
    for name in ("q_proj", "k_proj", "v_proj"):
        if _linear_quant_method_type(quant_config, f"{prefix}.{name}") is not fused:
            raise ValueError(
                f"{prefix}.{name} is not quantized like the fused "
                f"{prefix}.qkv_proj; list qkv_proj in the quantization "
                "config's ignore list together with q/k/v_proj."
            )
    index = {
        _linear_quant_method_type(quant_config, f"{prefix}.indexer.{name}")
        for name in ("linear_q_proj", "linear_k_proj")
    }
    if index == {fused}:
        return True
    if index != {UnquantizedLinearMethod}:
        raise ValueError(
            f"{prefix}.indexer projections are quantized unlike {prefix}.qkv_proj; "
            "only BF16 index projections can run apart from it."
        )
    return False


def _support_windows(config: PreTrainedConfig) -> tuple[int, int]:
    """Pages pinned at the front and at the query, in pages.

    The checkpoint states both windows in tokens. Without a local window the
    query's own page is still pinned, because a page the query only partially
    covers scores below pages it covers fully.
    """
    block = int(config.block_sparse_gqa_block_size)
    prefix_tokens = int(getattr(config, "block_sparse_gqa_prefix_window_size", 0))
    local_tokens = int(getattr(config, "block_sparse_gqa_local_window_size", 0))
    return prefix_tokens // block, max(local_tokens // block, 1)


class MaxwellV1FlashMLP(nn.Module):
    """Dense SwiGLU FFN, used for ``mlp_only_layers`` and the shared expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"unsupported activation {hidden_act!r}, expected 'silu'")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class MaxwellV1FlashGate(GateLinear):
    """Router projection through the split-BF16 GEMM.

    The weight is split once into a BF16 high part and a BF16 residual whose
    products accumulate in FP32 inside one kernel; two separate matmuls would
    round each back to BF16 and lose the residual. Falls back to
    ``GateLinear``'s dispatch until ``prepare_split_weights`` has run.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_w_hi", None, persistent=False)
        self.register_buffer("_w_lo", None, persistent=False)

    def prepare_split_weights(self) -> None:
        """Split the router weight once, so forward never does it per step.

        The loader's ``process_weights_after_loading`` only reaches modules
        with a quant method, so the model drives this from ``load_weights``.
        Idempotent.
        """
        if not self.is_unquantized:
            return
        self._w_hi, self._w_lo = split_router_weight(self.weight.data)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.nn.Parameter | None]:
        if self._w_hi is None or x.dtype is not torch.bfloat16:
            return super().forward(x)
        return self._return(torch.ops.vllm.split_bf16_gemm(x, self._w_hi, self._w_lo))


class MaxwellV1FlashMoE(nn.Module):
    """Sparse MoE block with a shared expert.

    Router logits stay fp32: the fp32 correction bias's per-expert spread is
    below bf16 resolution.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > config.num_experts:
            raise ValueError(
                f"tensor parallel size {tp_size} exceeds the number of experts "
                f"{config.num_experts}"
            )
        self.score_function = getattr(config, "router_score_function", "softmax")
        if self.score_function not in ("softmax", "sigmoid"):
            raise ValueError(
                f"unsupported router_score_function {self.score_function!r}"
            )
        self.top_k = config.num_experts_per_tok

        # The checkpoint stores the routing correction bias separately, in fp32.
        self.expert_bias: nn.Parameter | None = None
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32)
            )

        self.gate = MaxwellV1FlashGate(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert: MaxwellV1FlashMLP | None = None
        if getattr(config, "shared_expert_intermediate_size", 0):
            self.shared_expert = MaxwellV1FlashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_expert",
            )

        moe_kwargs: dict = {}
        if self.score_function == "sigmoid":
            moe_kwargs.update(
                scoring_func="sigmoid",
                e_score_correction_bias=self.expert_bias,
                renormalize=True,
            )
        else:
            moe_kwargs.update(
                renormalize=getattr(config, "norm_topk_prob", True),
                custom_routing_function=self._route,
            )
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            router_logits_dtype=self.gate.out_dtype,
            shared_experts=self.shared_expert,
            prefix=f"{prefix}.experts",
            **moe_kwargs,
        )

    def _route(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and normalize experts for a softmax-routed checkpoint."""
        logits = gating_output
        if self.expert_bias is not None:
            logits = logits + self.expert_bias
        topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
        topk_weights = torch.softmax(topk_logits.float(), dim=-1)
        return topk_weights, topk_ids.to(torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(hidden_states)
        out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return out.view(num_tokens, hidden_dim)


class MaxwellV1FlashAttention(nn.Module):
    """Dense paged attention with per-head QK norm and full RoPE."""

    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads, self.num_kv_heads = tp_head_counts(config, tp_size)
        self.head_dim = _head_dim(config)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.qk_layernorm = bool(getattr(config, "qk_layernorm", False))
        if self.qk_layernorm:
            self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={"rope_theta": config.rope_theta},
        )
        self.use_fused_norm_rope = self.qk_layernorm and _fused_norm_rope_supported(
            self.rotary_emb, self.head_dim
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_fused_norm_rope:
            q, k = fused_qk_rmsnorm_rope(
                q,
                k,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_layernorm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            if self.qk_layernorm:
                num_tokens = q.shape[0]
                q = self.q_layernorm(
                    q.reshape(num_tokens * self.num_heads, self.head_dim)
                ).reshape(num_tokens, self.q_size)
                k = self.k_layernorm(
                    k.reshape(num_tokens * self.num_kv_heads, self.head_dim)
                ).reshape(num_tokens, self.kv_size)
            # Norm first, then rotate: the checkpoint's own order.
            q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class MaxwellV1FlashSparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention with the page-selecting index branch.

    Owns the main and index projections, QK norms, RoPE and the backend wiring
    a generic ``Attention`` layer would provide. One fused kernel normalizes,
    rotates and caches both branches; the indexer selects pages into the shared
    top-k buffer that the sparse attend reads.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        index_q_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_output: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert topk_indices_buffer is not None
        self.attn_output = attn_output
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads, self.num_kv_heads = tp_head_counts(config, tp_size)
        self.head_dim = _head_dim(config)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.total_index_heads = int(config.block_sparse_gqa_indexer_n_heads)
        self.num_index_heads = self.num_kv_heads
        self.index_head_dim = int(config.block_sparse_gqa_indexer_head_dim)
        self.index_q_size = self.num_index_heads * self.index_head_dim
        block_size = int(config.block_sparse_gqa_block_size)

        # One GEMM for [q | k | v | index_q | index_k] when all five share a
        # weight format; a BF16 indexer under a quantized qkv_proj gets its
        # own BF16 GEMM for [index_q | index_k].
        self.fuse_index_proj = _fuse_index_projection(quant_config, prefix)
        proj_args = (
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_index_heads,
            self.index_head_dim,
        )
        self.qkv_proj = MaxwellV1FlashQKVParallelLinear(
            *proj_args,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            shards=FUSED_SHARDS if self.fuse_index_proj else MAIN_SHARDS,
        )
        if not self.fuse_index_proj:
            logger.info_once(
                "Maxwell-V1-Flash index projections are BF16 under a quantized "
                "qkv_proj; running them as a separate BF16 GEMM."
            )
            self.index_proj = MaxwellV1FlashQKVParallelLinear(
                *proj_args,
                bias=False,
                prefix=f"{prefix}.index_proj",
                shards=INDEX_SHARDS,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # The index branch rotates with the main branch's embedding.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={"rope_theta": config.rope_theta},
        )
        if not _fused_norm_rope_supported(self.rotary_emb, self.head_dim):
            raise ValueError(
                "block_sparse_gqa normalizes and rotates the main and index "
                "branches in one kernel, which rotates the full head with "
                "NeoX pairing; this checkpoint's RoPE does not."
            )

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        index_kv_dtype = indexer_kv_dtype(vllm_config)
        # Owns its KV-cache insert path, so it publishes the per-layer scale
        # attributes a generic Attention layer would.
        set_default_quant_scales(self, register_buffer=True)

        topk_blocks = _topk_blocks(config)
        init_blocks, local_blocks = _support_windows(config)
        self.topk_indices_buffer = topk_indices_buffer
        self.attn_backend = MaxwellV1FlashSparseBackend
        self.impl: MaxwellV1FlashSparseImpl = MaxwellV1FlashSparseImpl(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=block_size,
        )
        self.indexer = MaxwellV1FlashIndexer(
            num_kv_heads=self.num_kv_heads,
            num_index_heads=self.num_index_heads,
            index_head_dim=self.index_head_dim,
            topk_blocks=topk_blocks,
            sparse_block_size=block_size,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            group_reduction=str(
                getattr(config, "block_sparse_gqa_group_reduction", "max")
            ),
            prefix=self.layer_name,
            rms_norm_eps=config.rms_norm_eps,
            cache_config=cache_config,
            indexer_kv_dtype=index_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
            index_q_mxfp8=index_q_mxfp8,
        )
        if self.indexer.index_mxfp6:
            from vllm.models.maxwell_v1_flash.nvidia.ops.mxfp6 import (
                MXFP6_INDEX_KERNEL,
            )

            MXFP6_INDEX_KERNEL.register_warmup()
        index_format = "fp8" if "fp8" in index_kv_dtype else str(index_kv_dtype)
        scheduler_config = vllm_config.scheduler_config
        mbsa_kv_dtype = (
            self.impl.kv_cache_fp8_dtype
            if self.impl.use_fp8_kv
            else self.kv_cache_torch_dtype
        )
        mbsa_reason = self.impl.configure_mbsa_prefill(
            kv_dtype=mbsa_kv_dtype,
            output_dtype=vllm_config.model_config.dtype,
            index_format=index_format,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            share_route=self.indexer.share_route,
            score_sum=self.indexer.score_sum,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_seqs=scheduler_config.max_num_seqs,
        )
        if mbsa_reason is None:
            MBSA_PREFILL_KERNEL.register_warmup(
                index_format=index_format,
                limits=self.impl.mbsa_prefill_limits,
                kv_dtype=mbsa_kv_dtype,
                q_dtype=(
                    torch.float8_e4m3fn if self.impl.use_fp8_kv else torch.bfloat16
                ),
            )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        # Piecewise capture would otherwise record the layer's per-step
        # metadata reads into the graph.
        splitting_op = "vllm::maxwell_v1_flash_sparse_attention"
        if (
            compilation_config.splitting_ops is not None
            and splitting_op not in compilation_config.splitting_ops
        ):
            compilation_config.splitting_ops.append(splitting_op)
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

    def get_attn_backend(self) -> type[MaxwellV1FlashSparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Opaque op so Dynamo does not freeze the forward-context metadata into
        # trace-time constants. As a piecewise split point it writes in place:
        # a returned tensor would move between replays.
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.maxwell_v1_flash_sparse_attention(
            positions, hidden_states, output, self.layer_name
        )
        return output

    def _forward_impl(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        qkv, _ = self.qkv_proj(hidden_states)

        forward_context = get_forward_context()
        fwd_slot_mapping = forward_context.slot_mapping
        attn_metadata = forward_context.attn_metadata
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
            or not isinstance(attn_metadata, dict)
        ):
            # Memory-profiling run: the caches are not bound yet.
            self.impl.reserve_profile_workspace(qkv.shape[0], qkv.device)
            output.zero_()
            return

        if self.fuse_index_proj:
            q, k, v, raw_index_q, raw_index_k = qkv.split(
                [
                    self.q_size,
                    self.kv_size,
                    self.kv_size,
                    self.index_q_size,
                    self.index_head_dim,
                ],
                dim=-1,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            index_qk, _ = self.index_proj(hidden_states)
            raw_index_q, raw_index_k = index_qk.split(
                [self.index_q_size, self.index_head_dim], dim=-1
            )
        # The main cache is [B, Hkv, N, 2 * D]; the fused writer wants the two
        # sides as [B, N, Hkv, D] views. An FP8 cache is allocated as bytes.
        kv_cache = self.kv_cache
        if kv_cache.dtype == torch.uint8:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)
        index_cache = self.indexer.index_cache
        # The packed MXFP6 cache is flat bytes; the others are [slots, D].
        index_k_cache = (
            index_cache.kv_cache.view(-1)
            if self.indexer.index_mxfp6
            else index_cache.kv_cache.view(-1, self.index_head_dim)
        )
        index_q_mx, index_q_sf = self.indexer.index_q_mxfp8 or (None, None)
        q, index_q = torch.ops.vllm.maxwell_v1_flash_fused_qk_norm_rope_cache_index(
            q,
            k,
            v,
            raw_index_q,
            raw_index_k,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            fwd_slot_mapping[self.layer_name],
            fwd_slot_mapping[index_cache.prefix],
            key_cache,
            value_cache,
            index_k_cache,
            INDEX_K_FP8_SCALE if self.indexer.index_k_fp8 else None,
            # Padded graph-capture rows have no cache slot of their own; the
            # writer stops at the real tokens.
            attn_metadata[self.layer_name].num_actual_tokens,  # type: ignore[attr-defined]
            self.q_layernorm.variance_epsilon,
            self.indexer.q_layernorm.variance_epsilon,
            self.num_heads,
            self.num_kv_heads,
            self.num_index_heads,
            self.head_dim,
            q_out_dtype=torch.float8_e4m3fn if self.impl.use_fp8_kv else q.dtype,
            # Index Q is a cast, never a quantization: page ranking is
            # invariant to a positive rescale, and the score kernels need one
            # dtype on both sides. MXFP6 scores the MXFP8 rows instead.
            index_q_out_dtype=(
                None if self.indexer.index_mxfp6 else self.indexer.index_cache.dtype
            ),
            index_q_mx=index_q_mx,
            index_q_sf=index_q_sf,
        )

        attn_output = self._select_and_attend(
            q,
            index_q,
            torch.empty_like(q, dtype=hidden_states.dtype)
            if self.attn_output is None
            else self.attn_output[: q.shape[0]],
        )
        if self._o_proj_writes_output(output):
            # What o_proj computes here, straight into the op's output.
            torch.mm(attn_output, self.o_proj.weight.t(), out=output)
        else:
            output.copy_(self.o_proj(attn_output)[0])

    def _o_proj_writes_output(self, output: torch.Tensor) -> bool:
        # Attention is unsharded under EDP, so o_proj has no all-reduce.
        o_proj = self.o_proj
        return (
            o_proj.tp_size == 1
            and o_proj.bias is None
            and isinstance(o_proj.quant_method, UnquantizedLinearMethod)
            and output.dtype == o_proj.weight.dtype
            and output.is_contiguous()
            and not envs_vllm.VLLM_BATCH_INVARIANT
        )

    @eager_break_during_capture
    def _select_and_attend(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Eager break: both read per-request metadata and cannot be captured.
        # The selection crosses through the shared top-k buffer, not a Python
        # value. An MBSA prefill step routes its prefill tokens itself.
        mbsa_prefill_step = self.impl.mbsa_prefill_step(self)
        self.indexer(index_query, skip_prefill=mbsa_prefill_step is not None)
        return self.impl.forward(
            self,
            query,
            self.kv_cache,
            output,
            index_query=index_query,
            mbsa_prefill_step=mbsa_prefill_step,
        )


class HyperConnection(nn.Module):
    """mHC mixing parameters for one sub-layer.

    Kept in fp32: the mixing weights do not survive bf16.
    """

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        n = config.num_residual_streams
        mix_size = n * (n + 2)
        self.mapping_proj = nn.Linear(
            n * config.hidden_size, mix_size, bias=False, dtype=torch.float32
        )
        self.bias = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
        # The checkpoint stores three separate gate scales; load_weights packs
        # them into the (3,) tensor the operators take.
        self.alpha = nn.Parameter(
            torch.full((3,), config.mhc_init_gating_factor, dtype=torch.float32)
        )

    @property
    def fn(self) -> torch.Tensor:
        return self.mapping_proj.weight


class MaxwellV1FlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        force_dense: bool = False,
        index_q_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_output: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layer_id = extract_layer_index(prefix)
        self.layer_idx = layer_id
        self.num_hidden_layers = config.num_hidden_layers
        self.mhc = getattr(config, "enable_hyper_connections", False)

        # The MTP draft layer attends densely whatever layer_freq says.
        if not force_dense and _is_sparse_layer(config, layer_id):
            self.self_attn: nn.Module = MaxwellV1FlashSparseAttention(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                index_q_mxfp8=index_q_mxfp8,
                attn_output=attn_output,
            )
        else:
            self.self_attn = MaxwellV1FlashAttention(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if _is_moe_layer(config, layer_id):
            self.block_sparse_moe: nn.Module = MaxwellV1FlashMoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.block_sparse_moe = MaxwellV1FlashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc:
            self.n = config.num_residual_streams
            self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.self_attn_hyper_connection = HyperConnection(config)
            self.mlp_hyper_connection = HyperConnection(config)
            self.mhc_pre_op = MaxwellV1FlashMHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MaxwellV1FlashMHCFusedPostPreOp()
            self._register_mhc_warmup(config)

    def _register_mhc_warmup(self, config: PreTrainedConfig) -> None:
        """Precompile the mHC TileLang kernels for this checkpoint's epsilon
        placement, so the first request does not pay the JIT."""
        vllm_config = get_current_vllm_config()
        if not vllm_config.kernel_config.enable_jit_warmup:
            return
        from vllm.model_executor.kernels.mhc.tilelang_kernels import (
            _HC_PRENORM_GEMM_TILELANG_KERNEL,
            _MHC_FUSED_TILELANG_KERNEL,
            _MHC_POST_TILELANG_KERNEL,
        )
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        hidden_size = config.hidden_size
        include_pre_gemm_splits = is_deep_gemm_supported()
        EPS_OUTSIDE_BIG_FUSE_KERNEL.register_warmup(
            vllm_config,
            hidden_size=hidden_size,
            hc_mult=self.n,
            use_norm_weight=True,
            include_pre_gemm_splits=include_pre_gemm_splits,
            include_broadcast_splits=False,
            rms_eps=_MHC_EPS,
            hc_pre_eps=_MHC_EPS,
            hc_sinkhorn_eps=_MHC_EPS,
            hc_post_mult_value=_MHC_POST_MULT,
            sinkhorn_repeat=self.sinkhorn_iterations,
            norm_eps=(
                self.input_layernorm.variance_epsilon,
                self.post_attention_layernorm.variance_epsilon,
            ),
        )
        if not include_pre_gemm_splits:
            _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                vllm_config,
                hidden_size=hidden_size,
                hc_mult=self.n,
                n_out=self.n * (2 + self.n),
            )
        _MHC_POST_TILELANG_KERNEL.register_warmup(
            hidden_size=hidden_size,
            hc_mult=self.n,
        )
        _MHC_FUSED_TILELANG_KERNEL.register_warmup(
            vllm_config,
            hidden_size=hidden_size,
            hc_mult=self.n,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Run the layer standalone, materializing its own mHC post."""
        if not self.mhc:
            residual = hidden_states
            hidden_states = self.self_attn(
                positions, self.input_layernorm(hidden_states)
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.block_sparse_moe(
                self.post_attention_layernorm(hidden_states)
            )
            return residual + hidden_states
        deferred = self.forward_mhc(positions, hidden_states)
        return self.finish_mhc(deferred)

    def forward_mhc(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the layer, deferring its final mHC post to the next layer.

        ``hidden_states`` carries the streams flattened as
        ``[tokens, n * hidden]``, which is also how they cross a pipeline
        boundary; the mHC operators want them as ``[tokens, n, hidden]``.
        """
        residual = hidden_states.view(hidden_states.shape[0], self.n, -1)
        post, comb, layer_input = self._hc_pre(
            residual,
            self.self_attn_hyper_connection,
            self.input_layernorm,
        )
        return self._attn_and_mlp(positions, residual, post, comb, layer_input)

    def continue_mhc(
        self,
        positions: torch.Tensor,
        deferred: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse the previous layer's mHC post with this layer's mHC pre."""
        prev_out, prev_residual, prev_post, prev_comb = deferred
        residual, post, comb, layer_input = self._hc_fused_post_pre(
            prev_out,
            prev_residual,
            prev_post,
            prev_comb,
            self.self_attn_hyper_connection,
            self.input_layernorm,
        )
        return self._attn_and_mlp(positions, residual, post, comb, layer_input)

    def finish_mhc(
        self,
        deferred: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Materialize the last mHC post, back to flattened streams.

        The streams are contracted by the learned ``HCHead``, not averaged, so
        they stay separate until the end of the trunk.
        """
        mlp_out, residual, post, comb = deferred
        hidden_states = self.mhc_post_op(mlp_out, residual, post, comb)
        return hidden_states.reshape(hidden_states.shape[0], -1)

    def _attn_and_mlp(
        self,
        positions: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        layer_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out = self.self_attn(positions, layer_input)
        residual, post, comb, layer_input = self._hc_fused_post_pre(
            attn_out,
            residual,
            post,
            comb,
            self.mlp_hyper_connection,
            self.post_attention_layernorm,
        )
        return self.block_sparse_moe(layer_input), residual, post, comb

    def _hc_pre(
        self, residual: torch.Tensor, hc: HyperConnection, norm: RMSNorm
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mhc_pre_op(
            residual=residual,
            fn=hc.fn,
            hc_scale=hc.alpha,
            hc_base=hc.bias,
            rms_eps=_MHC_EPS,
            hc_pre_eps=_MHC_EPS,
            hc_sinkhorn_eps=_MHC_EPS,
            hc_post_mult_value=_MHC_POST_MULT,
            sinkhorn_repeat=self.sinkhorn_iterations,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
        )

    def _hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc: HyperConnection,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=hc.fn,
            hc_scale=hc.alpha,
            hc_base=hc.bias,
            rms_eps=_MHC_EPS,
            hc_pre_eps=_MHC_EPS,
            hc_sinkhorn_eps=_MHC_EPS,
            hc_post_mult_value=_MHC_POST_MULT,
            sinkhorn_repeat=self.sinkhorn_iterations,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
        )


class HCHead(nn.Module):
    """Learned n-stream to 1-stream contraction, in place of an average."""

    def __init__(self, config: PreTrainedConfig) -> None:
        super().__init__()
        n = config.num_residual_streams
        self.n = n
        self.hc_head_fn = nn.Parameter(
            torch.zeros(n, n * config.hidden_size, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.op = HCHeadOp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(
            x.view(x.shape[0], self.n, -1),
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            _MHC_EPS,
            _MHC_EPS,
        )


@support_torch_compile
class MaxwellV1FlashModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.mhc = getattr(config, "enable_hyper_connections", False)
        self.n = getattr(config, "num_residual_streams", 1) if self.mhc else 1

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # Top-k buffer shared by every sparse layer, at a stable address so the
        # selection survives cudagraph capture and replay. Token-major so the
        # indexer writes its native [token, head, topk] layout; the attend
        # transposes. Tokens are padded to a multiple of four to keep the head
        # stride int4-aligned for the CSR builder's vectorized loads.
        if _attention_variant(config) == VARIANT_BLOCK_SPARSE_GQA:
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            tp_size = get_tensor_model_parallel_world_size()
            # One route per local KV head, which is what the index branch
            # shards to.
            self.topk_indices_buffer = torch.empty(
                (max_tokens + 3) // 4 * 4,
                tp_head_counts(config, tp_size)[1],
                _topk_blocks(config),
                dtype=torch.int32,
            )
        else:
            self.topk_indices_buffer = None
        # Index Q in MXFP8, written by each sparse layer's fused QK launch and
        # read by its scorer before the next layer's launch, so one buffer
        # serves every layer. Allocated here so the KV budget sees it.
        self.index_q_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None
        if (
            self.topk_indices_buffer is not None
            and indexer_kv_dtype(vllm_config) == "mxfp6"
        ):
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            rows = 2 * self.topk_indices_buffer.shape[1]
            self.index_q_mxfp8 = (
                torch.empty((max_tokens, rows, 128), dtype=torch.uint8),
                torch.empty((max_tokens, rows), dtype=torch.int32),
            )
        # The sparse attend's output, read by o_proj in the same layer before
        # the next layer writes it: one buffer serves every layer.
        self.attn_output: torch.Tensor | None = None
        if self.topk_indices_buffer is not None:
            self.attn_output = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                tp_head_counts(config, tp_size)[0] * _head_dim(config),
                dtype=vllm_config.model_config.dtype,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MaxwellV1FlashDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                index_q_mxfp8=self.index_q_mxfp8,
                attn_output=self.attn_output,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.mhc:
            self.hc_head = HCHead(config)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * self.n
        )

        # The MTP draft reads the residual streams before hc_head collapses
        # them; forward keeps the latest step's in this stable buffer.
        spec_config = vllm_config.speculative_config
        self._mtp_hidden_buffer: torch.Tensor | None = None
        if (
            self.mhc
            and spec_config is not None
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
            and get_pp_group().is_last_rank
        ):
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hidden_size * self.n,
                dtype=vllm_config.model_config.dtype,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
            if self.mhc:
                hidden_states = hc_expand(hidden_states, self.n).reshape(
                    hidden_states.shape[0], -1
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        layers = self.layers[self.start_layer : self.end_layer]
        if self.mhc and layers:
            # Keep each layer's final mHC post live until the next layer
            # consumes it, so one kernel does that post and the next layer's
            # pre together instead of round-tripping the residual.
            deferred = layers[0].forward_mhc(positions, hidden_states)
            for layer in layers[1:]:
                deferred = layer.continue_mhc(positions, deferred)
            hidden_states = layers[-1].finish_mhc(deferred)
        else:
            for layer in layers:
                hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self._mtp_hidden_buffer is not None:
            self._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(hidden_states)
        if self.mhc:
            hidden_states = self.hc_head(hidden_states)
        return self.norm(hidden_states)


class MaxwellV1FlashForCausalLM(nn.Module, SupportsPP):
    """Maxwell-V1-Flash MoE, dense or ``block_sparse_gqa``."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        _validate_attention_variant(config, vllm_config)

        self.model = MaxwellV1FlashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual streams ``[max_num_batched_tokens, n * hidden]``
        for the MTP draft; valid after each target step."""
        return self.model._mtp_hidden_buffer

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Checkpoint experts use w1=gate, w2=down, w3=up.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def trunk_name(name: str) -> str | None:
            # The MTP draft loads its own mtp_layers.* from the same files.
            if name.startswith("mtp_layers."):
                return None
            return name

        return load_maxwell_weights(self, weights, trunk_name)


def load_maxwell_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    checkpoint_name: Callable[[str], str | None],
) -> set[str]:
    """Load checkpoint tensors into ``model``; return the parameters filled.

    ``checkpoint_name`` maps a checkpoint name to this model's parameter
    namespace, or to None to skip it. The trunk and the MTP draft share every
    other rule.
    """
    # A leading dot keeps `.q_proj` from also matching `linear_q_proj`. The
    # index projections land in their own GEMM when the layer has one, so
    # those entries come first.
    stacked_params_mapping: list[tuple[str, str, int | str | None]] = [
        (".index_proj", ".indexer.linear_q_proj", "index_q"),
        (".index_proj", ".indexer.linear_k_proj", "index_k"),
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".qkv_proj", ".indexer.linear_q_proj", "index_q"),
        (".qkv_proj", ".indexer.linear_k_proj", "index_k"),
        (".gate_up_proj", ".w1", 0),
        (".gate_up_proj", ".w3", 1),
        (".down_proj", ".w2", None),
    ]
    # The checkpoint keeps the three mHC gate scales apart; the operators
    # take them as one (3,) tensor.
    alpha_slots = {".alpha_pre": 0, ".alpha_post": 1, ".alpha_res": 2}

    # Checkpoint experts use w1=gate, w2=down, w3=up.
    expert_params_mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=model.config.num_experts,
    )
    params_dict = dict(model.named_parameters())
    loaded_params: set[str] = set()
    # A sparse checkpoint ships index projections for its dense layers
    # too. They are not part of the dense attention, so they are dropped
    # rather than loaded into a projection that has no room for them.
    index_prefixes = tuple(
        f"{module_name}.indexer."
        for module_name, module in model.named_modules()
        if isinstance(module, MaxwellV1FlashSparseAttention)
    )

    for ckpt_name, loaded_weight in weights:
        if ckpt_name.endswith("rotary_emb.inv_freq"):
            continue
        name = checkpoint_name(ckpt_name)
        if name is None:
            continue
        if ".indexer." in name and not name.startswith(index_prefixes):
            continue

        for suffix, slot in alpha_slots.items():
            if not name.endswith(suffix):
                continue
            mapped = name[: -len(suffix)] + ".alpha"
            if is_pp_missing_parameter(mapped, model):
                break
            params_dict[mapped].data[slot] = loaded_weight.reshape(()).to(torch.float32)
            loaded_params.add(mapped)
            break
        else:
            loaded = _load_projection(
                model,
                name,
                loaded_weight,
                stacked_params_mapping,
                expert_params_mapping,
                params_dict,
            )
            if loaded is not None:
                loaded_params.add(loaded)

    _check_expert_bias_precision(model)
    _prepare_router_split_weights(model)
    return loaded_params


def _load_projection(
    model: nn.Module,
    name: str,
    loaded_weight: torch.Tensor,
    stacked_params_mapping: list[tuple[str, str, int | str | None]],
    expert_params_mapping: list[tuple[str, str, int, str]],
    params_dict: dict[str, torch.nn.Parameter],
) -> str | None:
    """Load one checkpoint tensor; return the parameter it landed in."""
    # Routed experts share the w1/w2/w3 names with the dense and shared
    # MLPs, so they are matched first.
    if ".experts." in name:
        for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
            if weight_name not in name:
                continue
            mapped = name.replace(weight_name, param_name)
            if is_pp_missing_parameter(mapped, model) or mapped not in params_dict:
                return None
            param = params_dict[mapped]
            param.weight_loader(
                param,
                loaded_weight,
                mapped,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return mapped
        return None

    for param_name, weight_name, shard in stacked_params_mapping:
        if weight_name + "." not in name:
            continue
        mapped = name.replace(weight_name + ".", param_name + ".")
        if is_pp_missing_parameter(mapped, model):
            return None
        if mapped not in params_dict:
            continue
        param = params_dict[mapped]
        if shard is None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        else:
            param.weight_loader(param, loaded_weight, shard)
        return mapped

    if is_pp_missing_parameter(name, model):
        return None
    if name not in params_dict:
        logger.warning("Skipping unexpected weight: %s", name)
        return None
    param = params_dict[name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)
    return name


def _sparse_attention(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Run one sparse attention layer, resolved from the forward context.

    The layer is looked up at call time rather than captured, so the
    per-step slot mapping and attention metadata the body reads are the
    current step's.
    """
    self = get_forward_context().no_compile_layers[layer_name]
    self._forward_impl(positions, hidden_states, output)


def _sparse_attention_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="maxwell_v1_flash_sparse_attention",
    op_func=_sparse_attention,
    mutates_args=["output"],
    fake_impl=_sparse_attention_fake,
)


def _prepare_router_split_weights(model: nn.Module) -> None:
    """Split every router weight for the BF16 GEMM, once the load is done."""
    for module in model.modules():
        if isinstance(module, MaxwellV1FlashGate):
            module.prepare_split_weights()


def _check_expert_bias_precision(model: nn.Module) -> None:
    """Fail the load if the routing correction bias collapsed to one value.

    Its per-expert spread is below bf16 resolution, so a bf16 round trip
    silently disables it. Every layer is checked, since collapse can be partial.
    """
    for name, module in model.named_modules():
        bias = getattr(module, "expert_bias", None)
        if bias is None:
            continue
        if bias.dtype is not torch.float32:
            raise RuntimeError(
                f"{name}.expert_bias loaded as {bias.dtype}; the routing "
                "correction needs fp32 to keep its per-expert spread"
            )
        if bias.detach().unique().numel() <= 1:
            raise RuntimeError(
                f"{name}.expert_bias has one distinct value across "
                f"{bias.numel()} experts, so the routing correction does "
                "nothing. It was most likely rounded through bf16."
            )


def _validate_mxfp6_index(vllm_config: VllmConfig, kv_heads_per_rank: int) -> None:
    """Reject what the MXFP6 index scorers are not built for."""
    if not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype=mxfp6 (MAXWELL_MBSA_INDEX=mxfp6) scores with "
            "tcgen05 block-scaled MMAs and needs an SM100-family GPU; use fp8 "
            "or bf16 on this device."
        )
    from vllm.models.maxwell_v1_flash.nvidia.ops.mxfp6 import kernels

    limits = kernels()
    if kv_heads_per_rank != limits.INDEX_HEADS:
        raise ValueError(
            f"the MXFP6 index scorers are built for {limits.INDEX_HEADS} index "
            f"heads a rank, and this tensor-parallel size gives "
            f"{kv_heads_per_rank}; use fp8 or bf16."
        )

    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    if max_num_seqs > limits.MAX_DECODE_REQS:
        raise ValueError(
            f"the MXFP6 decode scorer schedules at most "
            f"{limits.MAX_DECODE_REQS} requests a step; got --max-num-seqs "
            f"{max_num_seqs}."
        )
    spec = vllm_config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    if 1 + (num_spec or 0) > limits.MAX_DECODE_QUERY_LEN:
        raise ValueError(
            f"the MXFP6 decode scorer takes at most "
            f"{limits.MAX_DECODE_QUERY_LEN} queries a request; got {num_spec} "
            "speculative tokens."
        )


def _validate_attention_variant(
    config: PreTrainedConfig, vllm_config: VllmConfig
) -> None:
    """Reject configurations this implementation cannot serve."""
    variant = _attention_variant(config)
    if variant is None:
        return
    if variant != VARIANT_BLOCK_SPARSE_GQA:
        raise ValueError(
            f"attention_variant {variant!r} is not implemented; expected "
            f"{VARIANT_BLOCK_SPARSE_GQA!r} or an unset value for a dense "
            "checkpoint."
        )
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    kv_heads_per_rank = tp_head_counts(config, tp_size)[1]
    if indexer_kv_dtype(vllm_config) == "mxfp6":
        _validate_mxfp6_index(vllm_config, kv_heads_per_rank)
    group_reduction = str(getattr(config, "block_sparse_gqa_group_reduction", "max"))
    # Also rejects an unrecognized reduction, whatever the TP size.
    share_route = page_reduction(group_reduction)[1]
    if tp_size > 1 and share_route:
        raise ValueError(
            f"block_sparse_gqa_group_reduction={group_reduction!r} routes every "
            "KV head together, which needs the index-head scores from all "
            "ranks; serve it with --tensor-parallel-size 1."
        )
    cache_dtype = str(vllm_config.cache_config.cache_dtype)
    if "fp8" in cache_dtype and cache_dtype not in ("fp8", "fp8_e4m3"):
        raise ValueError(
            f"--kv-cache-dtype {cache_dtype} is not supported: the fused cache "
            "writer quantizes K/V to FP8 E4M3, so use fp8 or fp8_e4m3."
        )

    block = int(config.block_sparse_gqa_block_size)
    topk = _topk_blocks(config)
    if topk not in SUPPORTED_TOPK_BLOCKS:
        raise ValueError(
            f"this checkpoint was trained to select {topk} pages, but the "
            f"sparse attend is instantiated for {SUPPORTED_TOPK_BLOCKS}. "
            "Serving any other count changes the trained attention."
        )
    index_head_dim = int(config.block_sparse_gqa_indexer_head_dim)
    if index_head_dim != 128:
        raise ValueError(
            "the sparse attend supports an index head dimension of 128, got "
            f"{index_head_dim}."
        )
    if index_head_dim != _head_dim(config):
        raise ValueError(
            f"the index head dimension ({index_head_dim}) must equal the "
            f"attention head dimension ({_head_dim(config)})."
        )
    if not getattr(config, "qk_layernorm", False):
        raise ValueError(
            "block_sparse_gqa needs qk_layernorm: one kernel normalizes and "
            "rotates the main and index branches together, and there is no "
            "un-normalized variant of it."
        )
    if int(config.block_sparse_gqa_indexer_n_heads) != config.num_key_value_heads:
        raise ValueError(
            "the indexer selects one page route per KV head, so it needs one "
            f"index head per KV head; got "
            f"{config.block_sparse_gqa_indexer_n_heads} index heads and "
            f"{config.num_key_value_heads} KV heads."
        )
    rotary_base = float(
        getattr(config, "block_sparse_gqa_indexer_rotary_base", config.rope_theta)
    )
    if rotary_base != float(config.rope_theta):
        raise ValueError(
            "the index branch shares the main branch's RoPE, but the indexer "
            f"base {rotary_base} differs from rope_theta {config.rope_theta}."
        )

    if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
        logger.info_once(
            "block_sparse_gqa with prefix caching: page selection follows "
            "prefill chunking, so outputs differ from a "
            "--no-enable-prefix-caching run."
        )

    init_blocks, local_blocks = _support_windows(config)
    dense_layers = [
        i for i in range(config.num_hidden_layers) if not _is_sparse_layer(config, i)
    ]
    logger.info(
        "Maxwell-V1-Flash block_sparse_gqa: %d pages of %d tokens (%d-token "
        "budget) = %d prefix + %d selected + %d local; reduce=%s; dense "
        "layers: %s",
        topk,
        block,
        topk * block,
        init_blocks,
        topk - init_blocks - local_blocks,
        local_blocks,
        getattr(config, "block_sparse_gqa_group_reduction", "max"),
        dense_layers or "none",
    )
