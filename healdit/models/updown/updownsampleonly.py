from typing import Tuple, List

import torch
import torch.nn as nn

from healdit.models.healparts import HEALDownSampler, HEALUpSampler, HEALEncoder, HEALDecoder
from healdit.batch import Batch
from healdit.utils.geo import get_lat_lon_flat_grid


class UpDownSampleOnly(nn.Module):

    def __init__(
            self,
            config,
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
        self.encoders = nn.Sequential()
        nfd = config.node_feat_dim
        running_n = config.starting_n
        for i, depth in enumerate(config.depths):
            self.encoders.append(
                HEALDownSampler(
                    rec=2 ** (running_n - 1),
                    send=2 ** running_n,
                    edge_in=config.edge_feat_dim, 
                    edge_out=config.edge_embed_dim,
                    lin_in=config.edge_embed_dim+nfd,
                    lin_out=2*nfd,
                )
            )
            nfd = nfd * 2
            running_n = running_n - 1


        self.heal_decoder = HEALDecoder(
            rec=(lon_flat, lat_flat),
            send=2**config.starting_n,
            edge_in=config.edge_feat_dim,
            edge_out=config.edge_embed_dim,
            lin_in=config.node_feat_dim + config.edge_embed_dim,
            lin_out=config.output_feat_dim,
        )

        nfd = config.node_feat_dim * (2 ** len(config.depths))
        running_n = config.starting_n - len(config.depths)
        self.decoders = nn.Sequential()
        for i, depth in enumerate(config.depths):
            self.decoders.append(
                HEALUpSampler(
                    rec=2 ** (running_n + 1),
                    send=2 ** running_n,
                    edge_in=config.edge_feat_dim, 
                    edge_out=config.edge_embed_dim,
                    lin_in=config.edge_embed_dim+nfd,
                    lin_out=nfd // 2,
                )
            )
            nfd = nfd // 2
            running_n = running_n + 1

    def forward(self, x: Batch) -> Tuple[Batch, List[torch.Tensor]]:
        var_names = list(x.data_vars.keys())

        x = self.heal_encoder(x.values)
        x = self.encoders(x)
        x = self.decoders(x)
        x = self.heal_decoder(x)

        x = Batch(data_vars=dict(zip(var_names, x.split(1, dim=-1))))

        return x

    def normalise(self, x: Batch) -> Batch:
        return x.normalise(self.normalisation)

    def unnormalise(self, x: Batch) -> Batch:
        return x.unnormalise(self.normalisation)