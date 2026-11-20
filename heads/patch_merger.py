from typing import Optional
import torch
import torch.nn as nn

from trace import TensorTrace


class PatchMerger(nn.Module):
    def __init__(self, hidden_size: int, layer_norm_eps: float, grid_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.grid_size = grid_size

    def forward(self, tokens: torch.Tensor, trace: Optional[TensorTrace] = None) -> torch.Tensor:
        class_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        if trace is not None:
            trace.log("merger.class_token", class_token, "CLIP class token")
            trace.log("merger.patch_tokens", patch_tokens, "one token per image patch")

        # Broadcasts input to the provided shape
        # or jsut let be the current shape and it still would be broadcasted
        class_token = torch.broadcast_to(class_token, patch_tokens.shape)
        merged = self.layer_norm(patch_tokens * class_token)
        # feature sgating mechanism, preserve the cls token without compute overhead
        # like appending extra channels, if dropping would lose some feature information
        if trace is not None:
            trace.log("merger.product", merged, "patch * class, then LayerNorm")

        batch, _, hidden = merged.shape
        feature_map = merged.reshape(batch, self.grid_size, self.grid_size, hidden)
        if trace is not None:
            trace.log("merger.feature_map", feature_map, "patches return to grid shape")
        
        return feature_map
