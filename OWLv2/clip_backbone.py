import torch
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel
from transformers.masking_utils import create_causal_mask

from trace import TensorTrace


class ClipBackbone(nn.Module):
    def __init__(self, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, projection_dim: int):
        super().__init__()
        self.vision = CLIPVisionModel(vision_config)
        self.text = CLIPTextModel(text_config)
        self.visual_projection = nn.Linear(vision_config.hidden_size, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_config.hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_image(self, pixel_values: torch.Tensor, trace: TensorTrace | None = None) -> torch.Tensor:
        if trace is not None:
            trace.section("CLIP vision tower  (transformers.CLIPVisionModel)")
            trace.log("vision.pixels", pixel_values, "preprocessed image")

        hidden = self.vision.embeddings(pixel_values)
        if trace is not None:
            trace.log("vision.embeddings", hidden, "[CLS] plus one token per 16x16 patch")

        hidden = self.vision.pre_layrnorm(hidden)
        if trace is not None:
            trace.log("vision.pre_layernorm", hidden)

        for index, layer in enumerate(self.vision.encoder.layers):
            hidden = layer(hidden, attention_mask=None)
            if trace is not None:
                trace.log(f"vision.layer[{index:02d}]", hidden, "LayerNorm -> attention -> residual -> MLP")

        tokens = self.vision.post_layernorm(hidden)
        if trace is not None:
            trace.log("vision.post_layernorm", tokens, "applied to every token, not only [CLS]")
        return tokens

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        trace: TensorTrace | None = None,
    ) -> torch.Tensor:
        if trace is not None:
            trace.section("CLIP text tower  (transformers.CLIPTextModel)")
            trace.log("text.input_ids", input_ids, "CLIP token ids, context length 16")

        hidden = self.text.embeddings(input_ids=input_ids)
        if trace is not None:
            trace.log("text.embeddings", hidden, "token embedding + position embedding")

        causal_mask = create_causal_mask(
            config=self.text.config,
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        for index, layer in enumerate(self.text.encoder.layers):
            hidden = layer(hidden, attention_mask=causal_mask)
            if trace is not None:
                trace.log(f"text.layer[{index:02d}]", hidden, "causal self-attention")

        hidden = self.text.final_layer_norm(hidden)
        if trace is not None:
            trace.log("text.final_layer_norm", hidden)

        eos_index = input_ids.to(dtype=torch.int).argmax(dim=-1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), eos_index]
        if trace is not None:
            trace.log("text.eos_pooled", pooled, "hidden state at the end-of-text token")

        projected = self.text_projection(pooled)
        if trace is not None:
            trace.log("text.projection", projected, "shared contrastive space")

        normalized = projected / torch.linalg.norm(projected, ord=2, dim=-1, keepdim=True)
        if trace is not None:
            trace.log("text.normalized", normalized, "these are the query embeddings")
        return normalized
