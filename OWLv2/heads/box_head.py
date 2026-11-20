from typing import Optional

import torch
import torch.nn as nn

from trace import TensorTrace


class BoxHead(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.dense0 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=out_dim)

    def forward(self, image_features: torch.Tensor, trace: Optional[TensorTrace] = None, name: str = "box_head") -> torch.Tensor:
        output = self.dense0(image_features)
        if trace is not None:
            trace.log(f"{name}.dense0", output, "linear, same num of channels")
        output = self.gelu(output)
        output = self.dense1(output)
        if trace is not None:
            trace.log(f"{name}.dense1", output, "linear, same num of channels")
        output = self.gelu(output)
        output = self.dense2(output)
        if trace is not None:
            trace.log(f"{name}.dense2", output, f"linear to {output.shape[-1]} values")
        return output

