"""Native Qwen3 token classifier with an optional explicit noncausal mask."""

import torch
from transformers import Qwen3ForTokenClassification


class PrivacyFilterModel(Qwen3ForTokenClassification):
    """Keep the Transformers classifier and backbone, changing only the mask.

    The attention mode is serialized in the Hugging Face config. A 4D additive
    mask makes Transformers' Qwen3 model bypass its internal causal-mask factory.
    """

    def __init__(self, config):
        mode = getattr(config, "privacy_attention_mode", "causal")
        if mode not in {"causal", "bidirectional"}:
            raise ValueError(f"Unsupported attention mode: {mode}")
        config.privacy_attention_mode = mode
        config.use_cache = False
        super().__init__(config)
        # Transformers infers this from the Python class name. Our subclass has
        # a different name, so select the native token loss explicitly.
        self.loss_type = "ForTokenClassification"
        if mode == "bidirectional":
            if self.config._attn_implementation not in {"eager", "sdpa"}:
                raise ValueError("Bidirectional attention requires eager or SDPA attention")
            for layer in self.model.layers:
                layer.self_attn.is_causal = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        **kwargs,
    ):
        if self.config.privacy_attention_mode == "bidirectional":
            if past_key_values is not None or use_cache:
                raise ValueError("KV cache is incompatible with bidirectional attention")
            inputs = input_ids if input_ids is not None else inputs_embeds
            if inputs is None:
                raise ValueError("input_ids or inputs_embeds is required")
            batch, length = inputs.shape[:2]
            if attention_mask is None:
                attention_mask = torch.ones((batch, length), device=inputs.device, dtype=torch.bool)
            if attention_mask.shape != (batch, length):
                raise ValueError("Expected a 2D padding mask matching the input")
            if not torch.all(attention_mask.bool().any(dim=-1)):
                raise ValueError("Each input must contain at least one non-padding token")
            dtype = self.model.embed_tokens.weight.dtype
            allowed = attention_mask.to(device=inputs.device, dtype=torch.bool)[:, None, None, :]
            attention_mask = torch.zeros((batch, 1, length, length), device=inputs.device, dtype=dtype)
            attention_mask = attention_mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            **kwargs,
        )
