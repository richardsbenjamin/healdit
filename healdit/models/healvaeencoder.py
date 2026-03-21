from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch 
import torch.nn as nn

from healdit.models.heal import HEALPix
from healdit.models.healparts import HEALDownSampler
from healdit.models.parts import FeedForward, MessagePassing, MLP
from healdit.utils.graph import (
    get_edge_features,
    get_edge_index,
    get_mesh_to_mesh_edge_index,
)

if TYPE_CHECKING:
    from typing import Tuple


class ResBlock(nn.Module):

    def __init__(
            self,
            node_feat_dim: int,
            node_hidden_dim: int,
            edge_embed_dim: int,
        ) -> None:
        super().__init__()
        self.feed_forward1 = nn.Sequential(
            nn.Linear(node_feat_dim, node_hidden_dim),
            nn.GELU()
        )
        self.message_passing1 = MessagePassing(node_hidden_dim, edge_embed_dim)
        self.message_passing2 = MessagePassing(node_hidden_dim, edge_embed_dim)
        self.feed_forward2 = nn.Linear(node_hidden_dim, node_feat_dim)

    def forward(self, x: torch.tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.tensor:
        residual = x
        x = self.feed_forward1(x)
        x = self.message_passing1(x, edge_index, edge_features)
        x = self.message_passing2(x, edge_index, edge_features)
        return residual + self.feed_forward2(x)


class HEALVAEEncoderBlock(nn.Module):

    def __init__(
            self,
            healpix: HEALPix,
            depth: int,
            node_feat_dim: int,
            node_hidden_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
            downsample: bool,       
        ) -> None:
        super().__init__()
        self.healpix = healpix
        self.depth = depth 
        self._set_res_block_edge_details()

        self.edge_embedder = MLP(
            in_dim=edge_feat_dim,
            hidden_dim=edge_embed_dim,
            out_dim=edge_embed_dim,
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                ResBlock(node_feat_dim, node_hidden_dim, edge_embed_dim)
            )
        self.downsample = nn.Identity() if not downsample else HEALDownSampler(
            rec=self.healpix.nside // 2,
            send=self.healpix.nside,
            embed_in=edge_feat_dim,
            embed_out=edge_embed_dim,
            lin_in=node_feat_dim+edge_embed_dim,
            lin_out=node_feat_dim*2,
        )

    def _set_res_block_edge_details(self) -> None:
        edge_index = get_mesh_to_mesh_edge_index(nside=self.healpix.nside)
        edge_attr = torch.from_numpy(
            get_edge_features(edge_index, rec=self.healpix.nside, send=self.healpix.nside)
            .astype(np.float32)
        )
        self.register_buffer("res_edge_index", edge_index)
        self.register_buffer("res_edge_attr", edge_attr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_embedder(self.res_edge_attr)
        for i, block in enumerate(self.blocks):
            x = block(x, self.res_edge_index, edge_features)
        return x, self.downsample(x)


class HEALVAEEncoder(nn.Module):

    def __init__(
            self,
            starting_n: int,
            depths: Tuple[int, ...],
            node_feat_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                HEALVAEEncoderBlock(
                    healpix=HEALPix(n=starting_n - i),
                    depth=depth,
                    node_feat_dim=node_feat_dim * (2 ** i),
                    node_hidden_dim=node_feat_dim * (2 ** i),
                    edge_feat_dim=edge_feat_dim,
                    edge_embed_dim=edge_embed_dim,
                    downsample=i != len(depths) - 1,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        for layer in self.layers:
            activation, x = layer(x)
            activations.append(activation)
        return activations

