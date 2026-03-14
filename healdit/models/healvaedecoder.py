from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch 
import torch.nn as nn

from healdit.models.heal import HEALPix
from healdit.models.healparts import HEALUpSampler
from healdit.models.healvaeencoder import ResBlock
from healdit.models.parts import MessagePassing, MLP
from healdit.utils.graph import (
    get_edge_features,
    get_mesh_to_mesh_edge_index,
)

if TYPE_CHECKING:
    from typing import List, Tuple

    from torch import Tensor


def draw_gaussian_diag_samples(mu: Tensor, logsigma: Tensor) -> Tensor:
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu
        
def gaussian_analytical_kl(
        mu1: Tensor,
        mu2: Tensor,
        logsigma1: Tensor,
        logsigma2: Tensor,
    ) -> Tensor:
    return (
        -0.5 + logsigma2 - logsigma1 
        + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
    )


class Block(nn.Module):

    def __init__(
            self,
            node_feat_dim: int,
            node_hidden_dim: int,
            edge_embed_dim: int,
            node_out_dim: int,
        ) -> None:
        super().__init__()
        self.feed_forward1 = nn.Sequential(
            nn.Linear(node_feat_dim, node_hidden_dim),
            nn.GELU()
        )
        self.message_passing1 = MessagePassing(node_hidden_dim, edge_embed_dim)
        self.message_passing2 = MessagePassing(node_hidden_dim, edge_embed_dim)
        self.feed_forward2 = nn.Linear(node_hidden_dim, node_out_dim)

    def forward(self, x: torch.tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.tensor:
        residual = x
        x = self.feed_forward1(x)
        x = self.message_passing1(x, edge_index, edge_features)
        x = self.message_passing2(x, edge_index, edge_features)
        return self.feed_forward2(x)


class TopDownBlock(nn.Module): 

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            z_dim: int,
            edge_dim: int,
        ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.block = Block(
            node_feat_dim=2*in_dim,
            node_hidden_dim=hidden_dim,
            node_out_dim=2*z_dim,
            edge_embed_dim=edge_dim,
        )
        self.prior = Block(
            node_feat_dim=in_dim,
            node_hidden_dim=hidden_dim,
            node_out_dim=2*z_dim+in_dim,
            edge_embed_dim=edge_dim,
        )
        self.z_feedforward = MLP(
            in_dim=z_dim, hidden_dim=in_dim, out_dim=in_dim,
        )
        self.res_out = ResBlock(
            node_feat_dim=in_dim,
            node_hidden_dim=hidden_dim,
            edge_embed_dim=edge_dim,
        )

    def forward(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: torch.Tensor,
        ) -> torch.Tensor:
        pfeat = self.prior(x, edge_index, edge_features)
        pm = pfeat[:, :, :self.z_dim]
        pv = pfeat[:, :, self.z_dim:self.z_dim*2]
        px = pfeat[:, :, self.z_dim*2:]

        xa = torch.cat([x, a], dim=-1)
        delta_m, delta_v = self.block(xa, edge_index, edge_features).chunk(2, dim=-1)
        qm = pm + delta_m
        qv = pv + delta_v

        z = self.z_feedforward(
            draw_gaussian_diag_samples(qm, qv)
        )
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        x = x + px 
        x = x + z
        x = x + self.res_out(x, edge_index, edge_features)
        return x, kl


class HEALVAEDecoderBlock(nn.Module):

    def __init__(
            self,
            healpix: HEALPix,
            node_feat_dim: int,
            node_hidden_dim: int,
            z_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
            depth: int,
            upsample: bool,
            n_edge_closest: int = 4,
        ) -> None:
        super().__init__()
        self.healpix = healpix
        self.n_edge_closest = n_edge_closest
        self._set_top_down_block_edge_details()
        self._set_upsampler_edge_details()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                TopDownBlock(
                    in_dim=node_feat_dim,
                    hidden_dim=node_hidden_dim,
                    z_dim=z_dim,
                    edge_dim=edge_embed_dim,
                )
            )
        self.edge_embedder = MLP(
            in_dim=edge_feat_dim,
            hidden_dim=edge_embed_dim,
            out_dim=edge_embed_dim,
        )
        self.upsample = nn.Identity() if not upsample else HEALUpSampler(
            edge_index=self.up_edge_index,
            edge_attr=self.up_edge_attr,
            embed_in=1,
            embed_out=edge_embed_dim,
            lin_in=(2*node_feat_dim)+edge_embed_dim,
            lin_out=node_feat_dim,
        )

    def _set_top_down_block_edge_details(self) -> None:
        edge_index = get_mesh_to_mesh_edge_index(nside=self.healpix.nside)
        edge_attr = torch.from_numpy(
            get_edge_features(edge_index, rec=self.healpix.nside, send=self.healpix.nside)
            .astype(np.float32)
        )
        self.register_buffer("top_down_edge_index", edge_index)
        self.register_buffer("top_down_edge_attr", edge_attr)

    def _set_upsampler_edge_details(self) -> None:
        healpix_down = HEALPix(n=self.healpix.n-1)
        edge_attr = (torch.arange(self.healpix.npix * self.n_edge_closest).to(torch.float32) % 4).reshape(-1, 1)
        edge_index = healpix_down.get_edge_index_by_knn(self.healpix, self.n_edge_closest)
        self.register_buffer("up_edge_index", edge_index)
        self.register_buffer("up_edge_attr", edge_attr)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        edge_features = self.edge_embedder(self.top_down_edge_attr)
        block_kl = []
        x = self.upsample(x)
        for block in self.blocks:
            x, kl = block(x, a, self.top_down_edge_index, edge_features)
            block_kl.append(kl)
        return x, block_kl


class HEALVAEDecoder(nn.Module):

    def __init__(
            self,
            starting_n: int,
            depths: Tuple[int],
            node_feat_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
            z_dim: int,
            n_edge_closest: int = 4,
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                HEALVAEDecoderBlock(
                    healpix=HEALPix(starting_n + i),
                    depth=depth,
                    node_feat_dim=int(node_feat_dim * (1 / (2 ** i))),
                    node_hidden_dim=int(node_feat_dim * (1 / (2 ** i))),
                    z_dim=z_dim,
                    edge_feat_dim=edge_feat_dim,
                    edge_embed_dim=edge_embed_dim,
                    upsample=i != 0,
                    n_edge_closest=n_edge_closest,
                )
            )

    def forward(self, activations: torch.Tensor) -> Tuple[torch.Tensor, List]:
        activations = activations[::-1]
        x = None
        decoder_kl = []
        for a, layer in zip(activations, self.layers): 
            if x is None:
                x = torch.zeros_like(a)
            x, layer_kl = layer(x, a)
            decoder_kl.extend(layer_kl)

        return x, decoder_kl
