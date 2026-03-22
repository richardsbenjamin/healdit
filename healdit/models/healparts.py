from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import torch 
import torch.nn as nn

from healdit.models.heal import HEALPix
from healdit.models.parts import FeedForward, FeedForwardSwin, MLP
from healdit.utils import get_attention_mask, scatter_sum
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
        send: Location,
        n_edge_closest: int,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
    r_lon, r_lat = resolve_location(rec)
    grid_vecs, _, _ = get_node_positions(r_lat, r_lon)

    # edge_attr = (torch.arange(len(r_lon) * n_edge_closest).to(dtype) % n_edge_closest).reshape(-1, 1)
    edge_index = HEALPix(nside=send).get_edge_index_by_knn(grid_vecs, n_edge_closest)
    edge_attr = torch.tensor(
        get_edge_features(edge_index.numpy(), send=send, rec=rec),
        dtype=dtype,
    )
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

        v_g = self.edge_embedder(v_g_prime)
        v_m_sum = scatter_sum(v_g, self.edge_index[1], dim=1)

        return self.g2m_linear(v_m_sum)


class HEALDecoder(nn.Module):

    def __init__(
            self,
            rec: Location,
            send: int,
            edge_in: int, 
            edge_out: int,
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
            in_dim=edge_in,
            hidden_dim=edge_out,
            out_dim=edge_out,
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
            rec: Location,
            send: Location,
            edge_in: int, 
            edge_out: int,
            lin_in: int,
            lin_out: int,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        self.dtype = dtype
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=edge_in,
            hidden_dim=edge_out,
            out_dim=edge_out,
        )
        self.linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: Location, send: Location) -> None:
        # npix_send = hp.nside2npix(2 ** send)
        # npix_rec = hp.nside2npix(2 ** rec)
        # edge_attr = torch.tensor(
        #     np.arange(npix_send) % (npix_send // npix_rec),
        #     dtype=torch.float32
        # ).reshape(-1, 1)

        edge_index = get_edge_index(send=send, rec=rec)
        edge_attr = torch.tensor(
            get_edge_features(edge_index.numpy(), send=send, rec=rec),
            dtype=self.dtype,
        )
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
            edge_in: int, 
            edge_out: int,
            lin_in: int,
            lin_out: int,
            n_edge_closest: int = 4,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__()
        self.n_edge_closest = n_edge_closest
        self.dtype = dtype
        self._init_edge_details(rec, send)
        self.edge_embedder = MLP(
            in_dim=edge_in,
            hidden_dim=edge_out,
            out_dim=edge_out,
        )
        self.linear = FeedForward(
            in_dim=lin_in,
            hidden_dim=lin_out,
            out_dim=lin_out,
        )

    def _init_edge_details(self, rec: Location, send: Location) -> None:
        healpix_send = HEALPix(nside=send)
        healpix_rec = HEALPix(nside=rec)
        # edge_attr = (torch.arange(hp.nside2npix(2 ** rec) * self.n_edge_closest).to(self.dtype) % self.n_edge_closest).reshape(-1, 1)
        edge_index = healpix_send.get_edge_index_by_knn(healpix_rec, self.n_edge_closest)
        edge_attr = torch.tensor(
            get_edge_features(edge_index.numpy(), send=send, rec=rec),
            dtype=self.dtype,
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)

    def forward(self, x):
        edge_features = self.edge_embedder(self.edge_attr)
        edge_features = edge_features.unsqueeze(0).expand(x.size(0), -1, -1)

        v_s = x[:, self.edge_index[0], :]

        v_s_prime = torch.cat([v_s, edge_features], dim=-1)
        v_m_sum = scatter_sum(v_s_prime, self.edge_index[1], dim=1)

        return self.linear(v_m_sum)









class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    """

    def __init__(self,
            in_features: int,
            window_size: int,
            number_of_heads: int,
            dropout_attention: float = 0.,
            dropout_projection: float = 0.,
            meta_network_hidden_features: int = 256,
            sequential_self_attention: bool = False,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.dtype = dtype
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(
            in_features=in_features, out_features=in_features * 3, bias=True, dtype=dtype,
            )
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True, dtype=dtype)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True, dtype=dtype))
        # Init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device, dtype=self.dtype)
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes]), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2) # .float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def _get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        return relative_position_bias.unsqueeze(0)

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self._get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
                                                             self.number_of_heads, tokens, tokens)
            attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens, tokens)
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Save original shape
        batch_size_windows, channels, height, width = input.shape  # type: int, int, int, int
        tokens: int = height * width
        # Reshape input to [batch size * windows, tokens (height * width), channels]
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        # Perform query, key, and value mapping
        query_key_value: torch.Tensor = self.mapping_qkv(input)
        query_key_value: torch.Tensor = query_key_value.view(batch_size_windows, tokens, 3, self.number_of_heads,
                                                             channels // self.number_of_heads).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]
        # Perform attention
        if self.sequential_self_attention:
            output: torch.Tensor = self.__sequential_self_attention(query=query, key=key, value=value,
                                                                    batch_size_windows=batch_size_windows,
                                                                    tokens=tokens,
                                                                    mask=mask)
        else:
            output: torch.Tensor = self.__self_attention(query=query, key=key, value=value,
                                                         batch_size_windows=batch_size_windows, tokens=tokens,
                                                         mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output

class HEALTransformerBlock(nn.Module):

    def __init__(self,
            hp_win: HEALWindow,  
            in_channels: int,
            number_of_heads: int,
            shift: bool = False,
            ff_feature_ratio: int = 4,
            dropout: float = 0.0,
            dropout_attention: float = 0.0,
            dropout_path: float = 0.0,
            sequential_self_attention: bool = False,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super(HEALTransformerBlock, self).__init__()
        self.hp_win = hp_win
        self.in_channels: int = in_channels
        self.shift = shift
        self.window_size = int(np.sqrt(self.hp_win.pix_per_win))

        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=in_channels, dtype=dtype)
        self.normalization_2: nn.Module = nn.LayerNorm(normalized_shape=in_channels, dtype=dtype)
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=in_channels,
            window_size=self.window_size,
            number_of_heads=number_of_heads,
            dropout_attention=dropout_attention,
            dropout_projection=dropout,
            sequential_self_attention=sequential_self_attention)
        self.dropout: nn.Module = timm.layers.DropPath(
            drop_prob=dropout_path) if dropout_path > 0. else nn.Identity()
        self.feed_forward_network: nn.Module = FeedForwardSwin(in_features=in_channels,
                                                           hidden_features=int(in_channels * ff_feature_ratio),
                                                           dropout=dropout,
                                                           out_features=in_channels,
                                                           )
        self._make_attention_mask()

    def _make_attention_mask(self) -> None:
        if self.shift:
            attention_mask = get_attention_mask(self.hp_win.shifted_windows_mask)
        else:
            attention_mask: Optional[torch.Tensor] = None

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, npix, channels = x.shape
        
        size = int(np.sqrt(self.hp_win.pix_per_win))

        if self.shift:
            output_patches = self.hp_win.shift_data(x)
        else:
            output_patches = x.unfold(dimension=1, size=size, step=size).permute(0, 2, 1, 3)

        output_patches = (
            output_patches.unfold(dimension=2, size=size, step=size)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, channels, size, size)
        )
        output_attention: torch.Tensor = self.window_attention(output_patches, mask=self.attention_mask)
        output_shift = output_attention.permute(0, 2, 3, 1).reshape(batch_size, -1, self.in_channels)

        if self.shift:
            output_shift = self.hp_win.unshift_data(output_shift)

        output_normalize: torch.Tensor = self.normalization_1(output_shift)
        output_skip: torch.Tensor = self.dropout(output_normalize) + x
        output_feed_forward: torch.Tensor = self.feed_forward_network(output_skip)
        output_normalize: torch.Tensor = self.normalization_2(output_feed_forward)
        output: torch.Tensor = output_skip + self.dropout(output_normalize)

        return output