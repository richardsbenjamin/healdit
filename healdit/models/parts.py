from __future__ import annotations

import torch
import torch.nn as nn

from healdit.utils import scatter_sum


class FeedForward(nn.Sequential):

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(
            nn.LayerNorm(in_dim, dtype=dtype),
            nn.Linear(in_features=in_dim, out_features=hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, dtype=dtype),
        )

class FeedForwardSwin(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features, dtype=torch.float32),
            nn.Dropout(p=dropout)
        )


class MessagePassing(nn.Module):
    
    def __init__(
            self,
            node_embed_dim: int,
            edge_embed_dim: int,
        ) -> None:
        super().__init__()
        self.mlp_e_m2m = MLP(
            in_dim=node_embed_dim*2 + edge_embed_dim,
            hidden_dim=edge_embed_dim,
            out_dim=edge_embed_dim,
        )
        self.mlp_vm_m2m = MLP(
            in_dim=node_embed_dim + edge_embed_dim,
            hidden_dim=node_embed_dim,
            out_dim=node_embed_dim,
        )

    def _update_mesh_edges(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: torch.Tensor,
        ) -> torch.Tensor:
        e_m2m = torch.cat([
            x[:, edge_index.T].reshape(x.shape[0], edge_index.shape[-1], -1),
            edge_features.expand(x.shape[0], -1, -1),
        ], dim=-1)
        return self.mlp_e_m2m(e_m2m)

    def _update_mesh_nodes(
            self,
            x: torch.Tensor,
            e_m2m_prime: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
        vm_m2m = torch.cat([
            x, scatter_sum(e_m2m_prime, edge_index[1], dim=1)
        ], dim=-1)
        return self.mlp_vm_m2m(vm_m2m)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_features: torch.Tensor,
        ) -> torch.Tensor:
        e_m2m_prime = self._update_mesh_edges(x, edge_index, edge_features)
        vm_m2m_prime = self._update_mesh_nodes(x, e_m2m_prime, edge_index)
        return x + vm_m2m_prime


class MLP(nn.Sequential):

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            dtype: torch.dtype = torch.float32
        ):
        super().__init__(
            nn.Linear(in_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, dtype=dtype)
        )

        