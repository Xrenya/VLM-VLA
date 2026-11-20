from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPTextConfig, CLIPVisionConfig, Owlv2Config

from clip_backbone import ClipBackbone
from heads.box_bias import grid_box_bias
from heads.box_head import BoxHead
from heads.class_head import ClassHead
from heads.patch_merger import PatchMerger
from trace import TensorTrace


@dataclass
class DetectionOutput:
    logits: torch.Tensor
    objectness_logits: torch.Tensor
    pred_boxes: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    class_embeds: torch.Tensor


def clip_config(config: Owlv2Config) -> Tuple[CLIPTextConfig, CLIPVisionConfig]:
    text = config.text_config
    vision = config.vision_config
    text_config = CLIPTextConfig(
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        intermediate_size=text.intermediate_size,
        num_hidden_layers=text.num_hidden_layers,
        num_attention_heads=text.num_attention_heads,
        max_position_embeddings=text.max_position_embeddings,
        hidden_act=text.hidden_act,
        layer_norm_eps=text.layer_norm_eps,
        attention_dropout=text.attention_dropout,
        initializer_range=text.initializer_range,
        initializer_factor=text.initializer_factor,
        pad_token_id=text.pad_token_id,
        bos_token_id=text.bos_token_id,
        attn_implementation="eager",

    )
    vision_config = CLIPVisionConfig(
        hidden_size=vision.hidden_size,
        intermediate_size=vision.intermediate_size,
        num_hidden_layers=vision.num_hidden_layers,
        num_attention_heads=vision.num_attention_heads,
        num_channels=vision.num_channels,
        image_size=vision.image_size,
        patch_size=vision.patch_size,
        hidden_act=vision.hidden_act,
        layer_norm_eps=vision.layer_norm_eps,
        attention_dropout=vision.attention_dropout,
        initializer_range=vision.initializer_range,
        initializer_factor=vision.initializer_factor,
        attn_implementation="eager",
    )
    return text_config, vision_config


class Owlv2Detector(nn.Module):
    def __init__(self, config: Owlv2Config) -> None:
        super().__init__()
        self.config = config
        text_config, vision_config = clip_config(config)
        vision = config.vision_config
        self.clip = ClipBackbone(text_config, vision_config, config.projection_dim)
        self.layer_norm_eps = vision.layer_norm_eps
        self.patch_size = vision.patch_size
        self.grid_size = vision.image_size // vision.patch_size

        self.class_head = ClassHead(vision_hidden_size=vision.hidden_size, text_hidden_size=text_config.hidden_size)
        self.box_head = BoxHead(hidden_size=vision.hidden_size, out_dim=4)
        self.objectness_head = BoxHead(hidden_size=vision.hidden_size, out_dim=1)
        self.patch_merger = PatchMerger(hidden_size=vision.hidden_size, layer_norm_eps=vision.layer_norm_eps, grid_size=self.grid_size)
        self.box_bias = nn.Buffer(grid_box_bias(self.grid_size, self.grid_size), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        trace: TensorTrace | None = None,
    ) -> DetectionOutput:
        tokens = self.clip.encode_image(pixel_values, trace=trace)
        query_embeds = self.clip.encode_text(input_ids, attention_mask=attention_mask, trace=trace)

        if trace is not None:
            trace.section("OWLv2 detection heads")

        feature_map = self.patch_merger(tokens, trace=trace)
        batch, grid_h, grid_w, hidden = feature_map.shape
        image_feats = feature_map.reshape(batch, grid_h * grid_w, hidden)

        if trace is not None:
            trace.log("image_feats", image_feats, "grid flattened to a list of patch tokens")

        if trace is not None:
            trace.log("query_embeds", query_embeds, "query tokens")

        num_queries = input_ids.shape[0] // batch

        query_embeds = query_embeds.reshape(batch, num_queries, query_embeds.shape[-1])
        query_ids = input_ids.reshape(batch, num_queries, input_ids.shape[-1])
        query_mask = query_ids[..., 0] > 0
        if trace is not None:
            trace.log("query_embeds", query_embeds, "one embedding per text query")
            trace.log("query_mask", query_mask, "False where the query was padding")
        
        logits, class_embeds = self.class_head(image_feats, query_embeds, query_mask, trace=trace)

        objectness_logits = self.objectness_head(image_feats.detach(), trace=trace, name="objectness_head")
        objectness_logits = objectness_logits[..., 0]
        if trace is not None:
            trace.log("objectness_logits", objectness_logits, "one score per patch, text is not used")

        pred_boxes = self.box_head(image_feats, trace=trace, name="box_head")
        pred_boxes = pred_boxes + self.box_bias.to(pred_boxes.device)
        if trace is not None:
            trace.log("box_head.plus_bias", pred_boxes, "MLP output plus the per-cell location prior")
        
        pred_boxes = torch.sigmoid(pred_boxes)
        if trace is not None:
            trace.log("pred_boxes", pred_boxes, "cx, cy, w, h in [0, 1]")        
        
        return DetectionOutput(
            logits=logits,
            objectness_logits=objectness_logits,
            pred_boxes=pred_boxes,
            text_embeds=query_embeds,
            image_embeds=feature_map,
            class_embeds=class_embeds,
        )


