from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Step:
    name: str
    shape: tuple
    dtype: str
    note: str


class TensorTrace:
    def __init__(self, enabled: bool = True):
        self.enabled: bool = enabled
        self.steps: List[Step] = []

    def log(self, name: str, tensor: torch.Tensor, note: str = "") -> torch.Tensor:
        step = Step(name, tensor.shape, dtype=str(tensor.dtype).replace("torch.", ""), note=note)
        self.steps.append(step)
        if self.enabled:
            details = f"  {note}" if note else ""
            print(f"  {name:<42} {str(step.shape):<28} {step.dtype}{details}")
        return tensor

    def section(self, title: str) -> None:
        self.steps.append(Step(name=title, shape=(), dtype="", note=""))
        if self.enabled:
            print(f"\n{title}")
