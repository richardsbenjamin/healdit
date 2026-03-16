from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import MISSING

if TYPE_CHECKING:
    from typing import Any, Tuple


@dataclass
class HEALVAEConfig:
    depths: Tuple[int, ...]
    edge_feat_dim: int
    edge_embed_dim: int
    input_feat_dim: int
    lat_lon_res: Tuple[int, int]
    node_feat_dim: int
    node_hidden_dim: int
    n_edge_closest: int
    output_feat_dim: int
    starting_n: int
    z_dim: int

@dataclass
class MSEParams:
    _target_: str = "torch.nn.MSELoss"
    reduction: str = "none"

@dataclass
class TrainParams:
    accumulation_steps: int
    batch_size: int
    data_path: str
    device: str
    epochs: int
    gradient_threshold: Optional[float]
    max_norm: float
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    criterion: Any = MISSING # nn.Module
    optimiser: Any = MISSING # optim.Optimizer


@dataclass
class Config:
    healvae: HEALVAEConfig = MISSING
    healvaetrainparams: TrainParams = MISSING