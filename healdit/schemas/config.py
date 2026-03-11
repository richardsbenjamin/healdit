from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple


@dataclass
class HEALVAEConfig:

    depths: Tuple[int, ...]
    edge_feat_dim: int
    edge_embed_dim: int
    input_feat_dim: int
    lon_lat_res: Tuple[int, int]
    node_feat_dim: int
    node_hidden_dim: int
    n_edge_closest: int
    output_feat_dim: int
    starting_n: int
    z_dim: int


@dataclass
class Config:
    healvae: HEALVAEConfig = MISSING
