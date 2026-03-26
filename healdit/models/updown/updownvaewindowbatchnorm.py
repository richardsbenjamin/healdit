from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

from healdit.batch import Batch
from healdit.models.heal import HEALPix, HEALWindow
from healdit.models.healparts import HEALDownSampler, HEALUpSampler, HEALEncoder, HEALDecoder, HEALTransformerBlock
from healdit.models.parts import FeedForward, MessagePassing, MLP
from healdit.utils.graph import (
    get_edge_features,
    get_edge_index,
    get_mesh_to_mesh_edge_index,
)

from healdit.utils.geo import get_lat_lon_flat_grid

if TYPE_CHECKING:
    from typing import Tuple


class ResBlock(nn.Module):

    def __init__(
            self,
            node_feat_dim: int,
            node_hidden_dim: int,
            num_heads: int, 
            n: int,
        ) -> None:
        super().__init__()
        self.feed_forward1 = nn.Sequential(
            nn.Linear(node_feat_dim, node_hidden_dim),
            nn.GELU()
        )
        self.message_passing1 = HEALTransformerBlock(
            hp_win=HEALWindow(n=n, w=1),  
            in_channels=node_hidden_dim,
            number_of_heads=num_heads,
            shift=False,
        )
        self.message_passing2 = HEALTransformerBlock(
            hp_win=HEALWindow(n=n, w=1),  
            in_channels=node_hidden_dim,
            number_of_heads=num_heads,
            shift=True,
        )
        self.feed_forward2 = nn.Linear(node_hidden_dim, node_feat_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        x = self.feed_forward1(x)
        x = self.message_passing1(x)
        x = self.message_passing2(x)
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
            num_heads: int,
            downsample: bool,       
        ) -> None:
        super().__init__()
        self.healpix = healpix
        self.depth = depth 

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                ResBlock(node_feat_dim, node_hidden_dim, num_heads, self.healpix.n)
            )
        self.downsample = nn.Identity() if not downsample else HEALDownSampler(
            rec=self.healpix.nside // 2,
            send=self.healpix.nside,
            edge_in=edge_feat_dim,
            edge_out=edge_embed_dim,
            lin_in=node_feat_dim+edge_embed_dim,
            lin_out=node_feat_dim*2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
        return x, self.downsample(x)


class HEALVAEEncoder(nn.Module):

    def __init__(
            self,
            starting_n: int,
            depths: Tuple[int, ...],
            node_feat_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
            num_heads: int,
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
                    num_heads=num_heads,
                    downsample=i != len(depths) - 1,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        for layer in self.layers:
            activation, x = layer(x)
            activations.append(activation)
        return activations


class Block(nn.Module):

    def __init__(
            self,
            node_feat_dim: int,
            node_hidden_dim: int,
            node_out_dim: int,
            num_heads: int,
            n: int,
        ) -> None:
        super().__init__()
        self.feed_forward1 = nn.Sequential(
            nn.Linear(node_feat_dim, node_hidden_dim),
            nn.GELU()
        )
        self.message_passing1 = HEALTransformerBlock(
            hp_win=HEALWindow(n=n, w=1),  
            in_channels=node_hidden_dim,
            number_of_heads=num_heads,
            shift=False,
        )
        self.message_passing2 = HEALTransformerBlock(
            hp_win=HEALWindow(n=n, w=1),  
            in_channels=node_hidden_dim,
            number_of_heads=num_heads,
            shift=True,
        )
        self.feed_forward2 = spectral_norm(nn.Linear(node_hidden_dim, node_out_dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        x = self.feed_forward1(x)
        x = self.message_passing1(x)
        x = self.message_passing2(x)
        return self.feed_forward2(x)


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


class TopDownBlock(nn.Module): 

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            z_dim: int,
            num_heads: int,
            n: int,
            use_bn: bool = True,
        ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.block = Block(
            node_feat_dim=2*in_dim,
            node_hidden_dim=hidden_dim,
            node_out_dim=2*z_dim,
            num_heads=num_heads,
            n=n,
        )
        self.use_bn = use_bn
        self.bn_qm = nn.BatchNorm1d(z_dim, momentum=0.01)
        self.bn_qm.bias.data.zero_()
        self.bn_qm.bias.requires_grad = False

        self.prior = Block(
            node_feat_dim=in_dim,
            node_hidden_dim=hidden_dim,
            node_out_dim=2*z_dim+in_dim,
            num_heads=num_heads,
            n=n,
        )
        self.z_feedforward = MLP(
            in_dim=z_dim, hidden_dim=in_dim, out_dim=in_dim,
        )
        self.res_out = ResBlock(
            node_feat_dim=in_dim,
            node_hidden_dim=hidden_dim,
            num_heads=num_heads,
            n=n,
        )

    def forward(
            self,
            x: torch.Tensor,
            a: torch.Tensor,
        ) -> torch.Tensor:
        pfeat = self.prior(x)
        pm = pfeat[:, :, :self.z_dim]
        pv = pfeat[:, :, self.z_dim:self.z_dim*2]
        px = pfeat[:, :, self.z_dim*2:]

        xa = torch.cat([x, a], dim=-1)
        delta_m, delta_v = self.block(xa).chunk(2, dim=-1)

        if self.use_bn:
            delta_m = delta_m.transpose(1, 2)
            delta_m = self.bn_qm(delta_m)
            delta_m = delta_m.transpose(1, 2)

        qm = pm + delta_m
        qv = pv + delta_v

        z = self.z_feedforward(
            draw_gaussian_diag_samples(qm, qv)
        )
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        x = x + px 
        x = x + z
        x = x + self.res_out(x)
        return x, kl

class HEALVAEDecoderBlock(nn.Module):

    def __init__(
            self,
            healpix: HEALPix,
            depth: int,
            node_feat_dim: int,
            node_hidden_dim: int,
            z_dim: int,
            edge_feat_dim: int,
            edge_embed_dim: int,
            num_heads: int,
            upsample: bool,   
            n_edge_closest: int = 4, 
            use_bn: bool = True,  
        ) -> None:
        super().__init__()
        self.healpix = healpix
        self.depth = depth 
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                TopDownBlock(
                    in_dim=node_feat_dim,
                    hidden_dim=node_hidden_dim,
                    z_dim=z_dim,
                    num_heads=num_heads,
                    n=self.healpix.n,
                    use_bn=use_bn,
                )
            )
        self.upsample = nn.Identity() if not upsample else HEALUpSampler(
            rec=self.healpix.nside,
            send=self.healpix.nside // 2,
            edge_in=edge_feat_dim,
            n_edge_closest=n_edge_closest,
            edge_out=edge_embed_dim,
            lin_in=2*node_feat_dim+edge_embed_dim,
            lin_out=node_feat_dim,
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        block_kl = []
        x = self.upsample(x)
        for block in self.blocks:
            x, kl = block(x, a)
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
            num_heads: int,
            z_dim: int,
            n_edge_closest: int = 4,
            use_bn: bool = True,
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                HEALVAEDecoderBlock(
                    healpix=HEALPix(n=starting_n + i),
                    depth=depth,
                    node_feat_dim=int(node_feat_dim * (1 / (2 ** i))),
                    node_hidden_dim=int(node_feat_dim * (1 / (2 ** i))),
                    edge_feat_dim=edge_feat_dim,
                    edge_embed_dim=edge_embed_dim,
                    num_heads=num_heads,
                    z_dim=z_dim,
                    upsample=i != 0,
                    n_edge_closest=n_edge_closest,
                    use_bn=use_bn,
                )
            )

    def forward(self, activations: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, List[torch.Tensor]]]:
        activations = activations[::-1]
        x = None
        decoder_kl = {}
        for a, layer in zip(activations, self.layers): 
            if x is None:
                x = torch.zeros_like(a)
            x, layer_kl = layer(x, a)
            decoder_kl[layer.healpix.n] = layer_kl

        return x, decoder_kl


class UpDownVAEWindowBatchNorm(nn.Module):

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
        self.encoder = HEALVAEEncoder(
            starting_n=config.starting_n,
            depths=config.depths,
            node_feat_dim=config.node_feat_dim,
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
            num_heads=config.num_heads
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

        self.decoder = HEALVAEDecoder(
            starting_n=last_n,
            depths=config.depths,
            node_feat_dim=last_node_dim,
            edge_feat_dim=config.edge_feat_dim,
            edge_embed_dim=config.edge_embed_dim,
            z_dim=config.z_dim,
            num_heads=config.num_heads,
            use_bn=config.use_bn,
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