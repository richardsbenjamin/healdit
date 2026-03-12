from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose

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
        overrides: Optional[List[str]] = None,
        schema_node: Optional[Node] = None,
    ) -> DictConfig:
    if overrides is None:
        overrides = sys.argv[1:]
    if schema_node is not None:
        cs = ConfigStore.instance()
        cs.store(name=config_name, node=schema_node)
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

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