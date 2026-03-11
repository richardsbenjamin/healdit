from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch 
import torch.nn as nn

from healdit.models.healparts import (
    HEALEncoder,
    HEALDecoder,
    get_encoder_edge_details,
    get_decoder_edge_details,
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
        self._set_encoder_decoder_edge_details()

        self.heal_encoder = HEALEncoder(
            edge_index=self.enc_edge_index,
            edge_attr=self.enc_edge_attr,
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
            edge_index=self.dec_edge_index,
            edge_attr=self.dec_edge_attr,
            embed_in=1,
            embed_out=config.edge_embed_dim,
            lin_in=config.node_feat_dim + config.edge_embed_dim,
            lin_out=config.output_feat_dim,
        )

    def _set_encoder_decoder_edge_details(self) -> None:
        lat_flat, lon_flat = get_lat_lon_flat_grid(*self.config.lat_lon_res)
        enc_edge_index, enc_edge_attr = get_encoder_edge_details(2 ** self.config.starting_n, lat_flat, lon_flat)
        dec_edge_index, dec_edge_attr = get_decoder_edge_details(
            self.config.starting_n, self.config.n_edge_closest, lat_flat, lon_flat,
        )
        self.register_buffer("enc_edge_index", enc_edge_index)
        self.register_buffer("enc_edge_attr", enc_edge_attr)
        self.register_buffer("dec_edge_index", dec_edge_index)
        self.register_buffer("dec_edge_attr", dec_edge_attr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.heal_encoder(x)
        activations = self.encoder(x)
        x, decoder_kl = self.decoder(activations)
        x = self.heal_decoder(x)
        return x, decoder_kl