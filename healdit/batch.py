from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Dict, List


def heal_collate_fn(batch_list: List[Batch]) -> Batch:
    variables = batch_list[0].data_vars.keys()
    
    batched_data = {}
    for v in variables:
        tensors = [b.data_vars[v] for b in batch_list]
        batched_data[v] = torch.stack(tensors, dim=0)
    
    return Batch(data_vars=batched_data)


@dataclass
class Batch:

    data_vars: Dict[str, torch.Tensor]

    def normalise(self, normalisation: dict) -> Batch:
        return Batch(
            data_vars={
                key: (value - normalisation[key]["mean"]) / normalisation[key]["std"] 
                for key, value in self.data_vars.items()
            }
        )

    def to(self, device: str) -> Batch:
        return Batch(
            data_vars={
                key: value.to(device) for key, value in self.data_vars.items()
            }
        )

    def unnormalise(self, normalisation: dict) -> Batch:
        return Batch(
            data_vars={
                key: (value * normalisation[key]["std"]) + normalisation[key]["mean"] 
                for key, value in self.data_vars.items()
            }
        )

    @property
    def values(self) -> torch.Tensor:
        return torch.concat(list(self.data_vars.values()), dim=-1)


class BatchLoss:

    def __init__(self, criterion: nn.Module) -> None:
        self.criterion = criterion

    def __call__(self, y: Batch, x: Batch) -> torch.Tensor:
        return self.criterion(y.values, x.values)
