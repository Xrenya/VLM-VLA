from typing import Optional, Tuple

import torch
import torch.nn as nn

from trace import TensorTrace


class ClassHead(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int) -> None:
        super().__init__()
        self.query_dim = vision_hidden_size
        self.dense0 = nn.Linear(in_features=vision_hidden_size, out_features=text_hidden_size)
        self.logit_shift = nn.Linear(in_features=vision_hidden_size, out_features=1)
        self.logit_scale = nn.Linear(in_features=vision_hidden_size, out_features=1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.Tensor,
        query_embeds: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        trace: Optional[TensorTrace] = None,
    ) -> Tuple[torch.Tensor]:
        # image_embeds - [batch, num_patches, vision_hidden]
        image_class_embeds = self.dense0(image_embeds)
        if trace is not None:
            trace.log("class_head.image_proj", image_class_embeds, "vision hidden -> text hidden")

        if query_embeds is None:
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros(
                (batch_size, num_patches, self.query_dim),
                device=image_class_embeds.device,
                dtype=image_class_embeds.dtype,
            )
            return pred_logits, image_class_embeds

        # query_embeds - [batch, num_queries, text_hidden]
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
        if trace is not None:
            trace.log("class_head.image_norm", image_class_embeds, "L2-normalized patch embeddings")
            trace.log("class_head.query_norm", query_embeds, "L2-normalized text embeddings")

        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        if trace is not None:
            trace.log("class_head.dot", pred_logits, "patch x query similarity")

        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.elu(self.logit_scale(image_embeds)) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale
        if trace is not None:
            trace.log("class_head.logits", pred_logits, "after per-patch shift and scale")

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = query_mask.unsqueeze(-2)
            pred_logits = torch.where(query_mask == 0, torch.finfo(pred_logits.dtype).min, pred_logits)
            pred_logits = pred_logits.to(torch.float32)
            if trace is not None:
                trace.log("class_head.masked_logits", pred_logits, "padded queries set to -inf")
            

        return pred_logits, image_class_embeds
