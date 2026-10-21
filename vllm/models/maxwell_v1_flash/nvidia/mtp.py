# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications for Maxwell-V1-Flash integration.
"""Maxwell-V1-Flash MTP draft using residual streams and shared target embeddings.

The draft receives pre-hc_head streams through get_mtp_target_hidden_states;
compute_logits collapses the streams and uses the target's shared LM head.
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.maxwell_v1_flash.nvidia.model import (
    HCHead,
    MaxwellV1FlashDecoderLayer,
    load_maxwell_weights,
)
from vllm.sequence import IntermediateTensors

_MTP_PREFIX = re.compile(r"^mtp_layers\.(\d+)\.")


def draft_weight_name(name: str, num_hidden_layers: int) -> str | None:
    """``mtp_layers.i.*`` -> ``model.layers.{num_hidden_layers + i}.*``.

    Everything else in the checkpoint belongs to the trunk, or is the shared
    embedding / lm_head, and returns None.
    """
    match = _MTP_PREFIX.match(name)
    if match is None:
        return None
    layer = num_hidden_layers + int(match.group(1))
    return f"model.layers.{layer}.{name[match.end() :]}"


# vLLM projection -> the checkpoint projections it fuses.
_FUSED_PROJECTIONS = {
    ".qkv_proj": (".q_proj", ".k_proj", ".v_proj"),
    ".gate_up_proj": (".w1", ".w3"),
    ".down_proj": (".w2",),
}


def draft_ignored_layers(ignored: Iterable[str], num_hidden_layers: int) -> set[str]:
    """Draft module names for the ``mtp_layers.*`` modules a checkpoint keeps
    unquantized.

    A quantized checkpoint lists them under their checkpoint names; a fused
    projection counts as unquantized when every projection it fuses is.
    """
    names: set[str] = set()
    for name in ignored:
        draft_name = draft_weight_name(name + ".", num_hidden_layers)
        if draft_name is not None:
            names.add(draft_name[:-1])
    for name in list(names):
        for fused, parts in _FUSED_PROJECTIONS.items():
            for part in parts:
                base = name.removesuffix(part)
                if base != name and all(base + p in names for p in parts):
                    names.add(base + fused)
    return names


class MaxwellV1FlashMTPLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.num_streams = config.num_residual_streams

        self.enorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            hidden,
            hidden,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            hidden,
            hidden,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.h_proj",
        )
        self.decoder_layer = MaxwellV1FlashDecoderLayer(
            config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.decoder_layer",
            force_dense=True,
        )
        self.hc_head = HCHead(config)
        self.final_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = inputs_embeds.shape[0]
        n, hidden = self.num_streams, self.hidden_size
        embeds = self.e_proj(self.enorm(inputs_embeds))
        streams = self.hnorm(
            previous_hidden_states[:num_tokens].reshape(num_tokens * n, hidden)
        )
        x = self.h_proj(streams).view(num_tokens, n, hidden) + embeds.unsqueeze(1)
        return self.decoder_layer(positions, x.reshape(num_tokens, n * hidden))

    def logits_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.final_layernorm(self.hc_head(hidden_states))


class MaxwellV1FlashMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.mtp_num_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleDict(
            {
                str(idx): MaxwellV1FlashMTPLayer(vllm_config, f"{prefix}.layers.{idx}")
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

    def layer(self, spec_step_idx: int) -> MaxwellV1FlashMTPLayer:
        idx = self.mtp_start_layer_idx + spec_step_idx % self.num_mtp_layers
        return self.layers[str(idx)]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layer(spec_step_idx)(
            positions, previous_hidden_states, inputs_embeds
        )


class MaxwellV1FlashMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        if not getattr(config, "enable_hyper_connections", False):
            raise ValueError("the Maxwell-V1-Flash MTP draft needs mHC streams")
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, Fp8Config) and quant_config.ignored_layers:
            ignored = quant_config.ignored_layers
            draft_ignored = draft_ignored_layers(ignored, config.num_hidden_layers)
            quant_config.ignored_layers = [*ignored, *sorted(draft_ignored)]
        self.model = MaxwellV1FlashMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        layer = self.model.layer(spec_step_idx)
        return self.logits_processor(self.lm_head, layer.logits_input(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        num_hidden_layers = self.config.num_hidden_layers
        # Everything under model.layers must come from the checkpoint, except
        # KV-cache scales; the embedding and lm_head are shared from the target
        # afterwards. Taken before loading: online quantization registers its
        # scales while the weights load.
        expected = [
            name
            for name, param in self.named_parameters()
            if name.startswith("model.layers.")
            and not isinstance(param, KVCacheScaleParameter)
        ]
        loaded = load_maxwell_weights(
            self, weights, lambda name: draft_weight_name(name, num_hidden_layers)
        )
        missing = sorted(name for name in expected if name not in loaded)
        if missing:
            raise ValueError(
                f"Maxwell-V1-Flash MTP draft: {len(missing)} parameters are not "
                f"in the checkpoint's mtp_layers.* tensors (first: {missing[:5]}). "
                "Serve a checkpoint with mtp_num_layers > 0, or disable "
                "speculative decoding."
            )
        return loaded
