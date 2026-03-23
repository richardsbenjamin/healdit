from healdit.batch import Batch
from healdit.utils.geo import get_lat_lon_flat_grid
import torch.nn as nn
import torch
from healdit.models.heal import HEALPix
from healdit.models.healparts import HEALDecoder, HEALEncoder
from healdit.models.healvaedecoder2 import HEALVAEDecoder2
from healdit.models.healvaeencoder import HEALVAEEncoder

from typing import Tuple, List

from healdit.batch import Batch
from healdit.utils.geo import get_lat_lon_flat_grid
import torch.nn as nn
import torch
from healdit.models.healparts import HEALDecoder, HEALEncoder, HEALDownSampler, HEALUpSampler

from typing import Tuple, List


class UpDown(nn.Module):

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
        self.encoders = HEALVAEEncoder(
            starting_n=config.starting_n,
            depths=config.depths,
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
        )

        self.heal_decoder = HEALDecoder(
            rec=(lon_flat, lat_flat),
            send=2**config.starting_n,
            edge_in=config.edge_feat_dim,
            edge_out=config.edge_embed_dim,
            lin_in=config.node_feat_dim + config.edge_embed_dim,
            lin_out=config.output_feat_dim,
        )

        last_node_dim = config.node_feat_dim * (2 ** (len(config.depths) - 1))
        last_n = config.starting_n - len(config.depths) + 1

        self.decoders = HEALVAEDecoder2(
            starting_n=last_n,
            depths=config.depths,
            node_feat_dim=last_node_dim,
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
        )

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
