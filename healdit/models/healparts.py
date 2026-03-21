from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import torch 
import torch.nn as nn

from healdit.models.heal import HEALPix
from healdit.models.parts import FeedForward, MLP
from healdit.utils import scatter_sum
from healdit.utils.graph import (
    get_edge_index,
    get_edge_features,
    get_node_positions,
    resolve_location,
)

if TYPE_CHECKING:
    from typing import Tuple

    from numpy import ndarray
    from torch import Tensor

    from healdit._typing import Location


def get_encoder_edge_details(
        rec: int,
        send: Location,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
    """Calculate the edge index and edge attributes for the encoder.

    Args:
        rec: Nside of receiving HEALPix grid.
        send: Nside or (lon, lat) of sending HEALPix grid.
        dtype: The data type of the edge attributes.

    Returns:
        A tuple containing the edge index and edge attributes.

    """
    edge_index = get_edge_index(send=send, rec=rec)
    edge_attr = torch.tensor(
        get_edge_features(edge_index.numpy(), send=send, rec=rec),
        dtype=dtype,
    )
    return edge_index, edge_attr

def get_decoder_edge_details(
        rec: Location,
        send: int,
        n_edge_closest: int,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
    r_lon, r_lat = resolve_location(rec)
    grid_vecs, _, _ = get_node_positions(r_lat, r_lon)

    edge_attr = (torch.arange(len(r_lon) * n_edge_closest).to(dtype) % n_edge_closest).reshape(-1, 1)
    edge_index = HEALPix(n=send).get_edge_index_by_knn(grid_vecs, n_edge_closest)
    return edge_index, edge_attr


class HEALEncoder(nn.Module):
    
    def __init__(
            self,
            rec: Location,
            send: Location,
            edge_in: int,
            edge_out: int,
            lin_in: int,
            lin_out: int,
        ) -> None:
        super().__init__()
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=edge_in,
            hidden_dim=edge_out,
            out_dim=edge_out,
        )
        self.g2m_linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: Location, send: Location) -> None:
        edge_index, edge_attr = get_encoder_edge_details(rec=rec, send=send)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_attr = self.edge_attr.unsqueeze(0).expand(x.size(0), -1, -1)
        v_g_prime = torch.cat([edge_attr, x], dim=-1)

        print(v_g_prime.shape)
        print(edge_attr.shape)
        print(x.shape)

        v_g = self.edge_embedder(v_g_prime)
        v_m_sum = scatter_sum(v_g, self.edge_index[1], dim=1)

        return self.g2m_linear(v_m_sum)


class HEALDecoder(nn.Module):

    def __init__(
            self,
            rec: Location,
            send: int,
            embed_in: int, 
            embed_out: int,
            lin_in: int,
            lin_out: int,
            n_edge_closest: int = 4,
            dtype=torch.float32
        ) -> None:
        super().__init__()
        self.n_edge_closest = n_edge_closest
        self.dtype = dtype
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=embed_in,
            hidden_dim=embed_out,
            out_dim=embed_out,
        )
        self.g2m_linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: Location, send: int) -> None:
        edge_index, edge_attr = get_decoder_edge_details(
            rec=rec, send=send, n_edge_closest=self.n_edge_closest, dtype=self.dtype
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

    def forward(self, x):
        edge_features = self.edge_embedder(self.edge_attr)
        edge_features = edge_features.unsqueeze(0).expand(x.size(0), -1, -1)

        v_s = x[:, self.edge_index[0], :]

        v_s_prime = torch.cat([v_s, edge_features], dim=-1)
        v_m_sum = scatter_sum(v_s_prime, self.edge_index[1], dim=1)

        return self.g2m_linear(v_m_sum)

        
class HEALDownSampler(nn.Module):

    def __init__(
            self,
            rec: int,
            send: int,
            embed_in: int, 
            embed_out: int,
            lin_in: int,
            lin_out: int,
            dtype=torch.float32
        ) -> None:
        super().__init__()
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=embed_in,
            hidden_dim=embed_out,
            out_dim=embed_out,
        )
        self.linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: int, send: int) -> None:
        npix_send = hp.nside2npix(2 ** send)
        npix_rec = hp.nside2npix(2 ** rec)
        edge_attr = torch.tensor(
            np.arange(npix_send) % (npix_send // npix_rec),
            dtype=torch.float32
        ).reshape(-1, 1)
        edge_index = get_edge_index(send=2 ** send, rec=2 ** rec)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

    def forward(self, x):
        edge_features = self.edge_embedder(self.edge_attr)
        edge_features = edge_features.unsqueeze(0).expand(x.size(0), -1, -1)

        v_m_prime = torch.cat([edge_features, x], dim=-1)
        x = scatter_sum(v_m_prime, self.edge_index[1], dim=1)

        return self.linear(x)


class HEALUpSampler(nn.Module):

    def __init__(
            self,
            rec: int,
            send: int,
            embed_in: int, 
            embed_out: int,
            lin_in: int,
            lin_out: int,
            n_edge_closest: int = 4,
            dtype=torch.float32
        ) -> None:
        super().__init__()
        self.n_edge_closest = n_edge_closest
        self.dtype = dtype
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=embed_in,
            hidden_dim=embed_out,
            out_dim=embed_out,
        )
        self.linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: int, send: int) -> None:
        healpix_send = HEALPix(n=send)
        healpix_rec = HEALPix(n=rec)
        edge_attr = (torch.arange(hp.nside2npix(2 ** rec) * self.n_edge_closest).to(self.dtype) % self.n_edge_closest).reshape(-1, 1)
        edge_index = healpix_send.get_edge_index_by_knn(healpix_rec, self.n_edge_closest)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

    def forward(self, x):
        edge_features = self.edge_embedder(self.edge_attr)
        edge_features = edge_features.unsqueeze(0).expand(x.size(0), -1, -1)

        v_s = x[:, self.edge_index[0], :]

        v_s_prime = torch.cat([v_s, edge_features], dim=-1)
        v_m_sum = scatter_sum(v_s_prime, self.edge_index[1], dim=1)

        return self.linear(v_m_sum)
