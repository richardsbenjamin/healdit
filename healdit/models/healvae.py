from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch 
import torch.nn as nn

from healdit.batch import Batch
from healdit.models.healparts import (
    HEALEncoder,
    HEALDecoder,
)
from healdit.models.healvaeencoder import HEALVAEEncoder
from healdit.models.healvaedecoder import HEALVAEDecoder
from healdit.utils.geo import get_lat_lon_flat_grid


if TYPE_CHECKING:
    from typing import Tuple

    from healdit.schemas.config import HEALVAEConfig


class HEALVAE(nn.Module):

    def __init__(
            self,
            config: HEALVAEConfig,
        ) -> None:
        super().__init__()
        # TO DO: validation checks e.g. if starting_n with # of depths makes sense
        self.config = config
        self.normalisation = config.normalisation["variables"]
        lat_flat, lon_flat = get_lat_lon_flat_grid(*self.config.lat_lon_res)
        
        self.heal_encoder = HEALEncoder(
            rec=2 ** config.starting_n,
            send=(lon_flat, lat_flat),
            edge_in=config.input_feat_dim+config.edge_feat_dim,
            edge_out=config.edge_embed_dim,
            lin_in=config.edge_embed_dim,
            lin_out=config.node_feat_dim,
        )
        self.encoder = HEALVAEEncoder(
            starting_n=config.starting_n,
            depths=config.depths,
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
        )
        self.decoder = HEALVAEDecoder(
            depths=config.depths,
            starting_n=config.starting_n - len(config.depths) + 1,
            node_feat_dim=config.node_feat_dim * (2 ** (len(config.depths) - 1)),
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
            z_dim=config.z_dim,
        )
        self.heal_decoder = HEALDecoder(
            rec=(lon_flat, lat_flat),
            send=config.starting_n,
            n_edge_closest=config.n_edge_closest,
            embed_in=1,
            embed_out=config.edge_embed_dim,
            lin_in=config.node_feat_dim + config.edge_embed_dim,
            lin_out=config.output_feat_dim,
        )

    def forward(self, x: Batch) -> Tuple[Batch, List[torch.Tensor]]:
        var_names = list(x.data_vars.keys())

        x = self.heal_encoder(x.values)
        activations = self.encoder(x)
        x, decoder_kl = self.decoder(activations)
        x = self.heal_decoder(x)

        x = Batch(data_vars=dict(zip(var_names, x.split(1, dim=-1))))

        return decoder_kl, x

    def normalise(self, x: Batch) -> Batch:
        return x.normalise(self.normalisation)

    def unnormalise(self, x: Batch) -> Batch:
        return x.unnormalise(self.normalisation)

