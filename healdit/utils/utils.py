from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch
from hydra.core.config_store import ConfigStore
from hydra import initialize, initialize_config_dir, compose

if TYPE_CHECKING:
    from typing import Optional, List

    from omegaconf import DictConfig
    from torch import Tensor
    from hydra.types import Node


def broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def load_config(
        config_name: str = "config", 
        config_path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
    if config_path and os.path.isabs(config_path):
        with initialize_config_dir(version_base=None, config_dir=config_path):
            return compose(config_name=config_name, overrides=overrides)

    path = config_path or "../config"
    with initialize(version_base=None, config_path=path):
        return compose(config_name=config_name, overrides=overrides)

def scatter_sum(
        src: Tensor,
        index: Tensor,
        dim: int = -1,
        out: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)