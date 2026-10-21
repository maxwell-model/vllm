# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.

from transformers import PreTrainedConfig


class MaxwellV1FlashConfig(PreTrainedConfig):
    """Configuration for the Maxwell-V1-Flash MoE model.

    Carried here so a checkpoint loads without ``--trust-remote-code``.
    Defaults are the trained values; the fields that decide the sparse
    attention have no safe default, so a checkpoint that omits one is served
    with the geometry it was trained on rather than a guess.
    """

    model_type = "maxwell_v1_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 200704,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        attention_head_size: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 524288,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 2.0e7,
        qk_layernorm: bool = True,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        attention_dropout: float = 0.0,
        # Mixture of experts. A layer in ``mlp_only_layers`` runs a dense FFN.
        num_experts: int = 256,
        num_experts_per_tok: int = 12,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 1024,
        mlp_only_layers: list[int] | None = None,
        norm_topk_prob: bool = False,
        router_score_function: str = "sigmoid",
        moe_router_enable_expert_bias: bool = True,
        # Multi-stream hyper-connections, in place of a plain residual.
        enable_hyper_connections: bool = True,
        num_residual_streams: int = 4,
        mhc_sinkhorn_iterations: int = 20,
        mhc_init_gating_factor: float = 0.01,
        # Block-sparse attention. Windows are stated in tokens; the indexer
        # selects ``block_sparse_gqa_total_support_blocks`` pages per query,
        # those windows included.
        attention_variant: str | None = "block_sparse_gqa",
        block_sparse_gqa_block_size: int = 128,
        block_sparse_gqa_topk_blocks: int = 16,
        block_sparse_gqa_total_support_blocks: int = 32,
        block_sparse_gqa_prefix_window_size: int = 1024,
        block_sparse_gqa_local_window_size: int = 1024,
        block_sparse_gqa_indexer_n_heads: int = 4,
        block_sparse_gqa_indexer_head_dim: int = 128,
        block_sparse_gqa_indexer_rotary_base: float | None = None,
        block_sparse_gqa_group_reduction: str = "per_group_max",
        block_sparse_gqa_layer_freq: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.qk_layernorm = qk_layernorm
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.mlp_only_layers = mlp_only_layers or []
        self.norm_topk_prob = norm_topk_prob
        self.router_score_function = router_score_function
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias

        self.enable_hyper_connections = enable_hyper_connections
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_init_gating_factor = mhc_init_gating_factor

        self.attention_variant = attention_variant
        self.block_sparse_gqa_block_size = block_sparse_gqa_block_size
        self.block_sparse_gqa_topk_blocks = block_sparse_gqa_topk_blocks
        self.block_sparse_gqa_total_support_blocks = (
            block_sparse_gqa_total_support_blocks
        )
        self.block_sparse_gqa_prefix_window_size = block_sparse_gqa_prefix_window_size
        self.block_sparse_gqa_local_window_size = block_sparse_gqa_local_window_size
        self.block_sparse_gqa_indexer_n_heads = block_sparse_gqa_indexer_n_heads
        self.block_sparse_gqa_indexer_head_dim = block_sparse_gqa_indexer_head_dim
        # The index branch rotates with the main branch's RoPE, so it shares
        # its base unless the checkpoint says otherwise.
        self.block_sparse_gqa_indexer_rotary_base = (
            rope_theta
            if block_sparse_gqa_indexer_rotary_base is None
            else block_sparse_gqa_indexer_rotary_base
        )
        self.block_sparse_gqa_group_reduction = block_sparse_gqa_group_reduction
        self.block_sparse_gqa_layer_freq = block_sparse_gqa_layer_freq

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MaxwellMoeConfig(MaxwellV1FlashConfig):
    """The same configuration under the name the released checkpoint uses."""

    model_type = "maxwell_moe"
