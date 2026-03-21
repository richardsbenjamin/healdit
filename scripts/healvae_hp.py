# import pickle

# import numpy as np
# import torch
# import xarray as xr
# from hydra.utils import instantiate

# from healdit.batch import heal_collate_fn
# from healdit.datasets import ZarrDataset
# from healdit.models import HEALVAE
# from healdit.train import train
# from healdit.utils import load_config
# from healdit.utils.parsers import comma_list_to_list, get_train_args


# def get_idx(ds: xr.Dataset, date: str) -> int:
#     return (ds.valid_time.values >= np.datetime64(date)).argmax()

# if __name__ == "__main__":
#     args = get_train_args()

#     cfg = load_config(args.config_name, overrides=comma_list_to_list(args.overrides))
#     healvae_cfg = instantiate(cfg.healvae)
#     train_params = instantiate(cfg.trainparams)
#     paths = instantiate(cfg.paths)

#     # Train split
#     ds = xr.open_zarr(paths.input_data_path, consolidated=True)
#     train_start_idx = get_idx(ds, train_params.train_start)
#     train_end_idx = get_idx(ds, train_params.train_end)
#     val_start_idx = get_idx(ds, train_params.val_start)
#     val_end_idx = get_idx(ds, train_params.val_end)
#     ds.close()

#     train_dataset = ZarrDataset(paths.input_data_path, time_slice=slice(train_start_idx, train_end_idx))
#     val_dataset = ZarrDataset(paths.input_data_path, time_slice=slice(val_start_idx, val_end_idx))

#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_params.batch_size, collate_fn=heal_collate_fn)
#     val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_params.batch_size, collate_fn=heal_collate_fn)

#     heal_vae = HEALVAE(healvae_cfg)
#     heal_vae = heal_vae.to(train_params.device)

#     if train_params.weight_init is not None:
#         train_params.weight_init(heal_vae, healvae_cfg)

#     train_history = train(
#         model=heal_vae,
#         loader=train_dataloader,
#         params=train_params,
#     )

#     with open(paths.history_path, "wb") as f:
#         pickle.dump(train_history, f)

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
        self.encoders = nn.Sequential()
        for i, depth in enumerate(config.depths):
            nfd = config.node_feat_dim * (2 ** i)
            self.encoders.append(
                HEALEncoder(
                    rec=2 ** (config.starting_n - i - 1),
                    send=2 ** (config.starting_n - i),
                    edge_in=nfd+config.edge_feat_dim,
                    edge_out=config.edge_embed_dim,
                    lin_in=config.edge_embed_dim,
                    lin_out=nfd ** 2,

                )
            )

        self.heal_decoder = HEALDecoder(
            rec=(lon_flat, lat_flat),
            send=config.starting_n,
            embed_in=1,
            embed_out=config.edge_embed_dim,
            lin_in=config.node_feat_dim + config.edge_embed_dim,
            lin_out=config.output_feat_dim,
        )

        last_node_dim = config.node_feat_dim * (2 ** (len(config.depths) - 1))
        last_n = config.starting_n - len(config.depths)

        self.decoders = nn.Sequential()
        for i, depth in enumerate(config.depths):
            nfd = int(last_node_dim * (1 / (2 ** i)))
            self.decoders.append(
                HEALDecoder(
                    rec=2 ** (last_n + i + 1),
                    send=last_n + i,
                    n_edge_closest=config.n_edge_closest,
                    embed_in=1,
                    embed_out=config.edge_embed_dim,
                    lin_in=nfd+config.edge_embed_dim,
                    lin_out=nfd,
                )
            )

    def forward(self, x: Batch) -> Tuple[Batch, List[torch.Tensor]]:
        var_names = list(x.data_vars.keys())

        print('X IN', x.shape)
        x = self.heal_encoder(x.values)
        print('X HEAL ENCODER', x.shape)
        x = self.encoders(x)
        print('X ENCODERS', x.shape)
        x = self.decoders(x)
        print('X DECODERS', x.shape)
        x = self.heal_decoder(x)
        print('X HEAL DECODER', x.shape)

        x = Batch(data_vars=dict(zip(var_names, x.split(1, dim=-1))))

        return x

    def normalise(self, x: Batch) -> Batch:
        return x.normalise(self.normalisation)

    def unnormalise(self, x: Batch) -> Batch:
        return x.unnormalise(self.normalisation)

import pickle

import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate

from healdit.batch import heal_collate_fn
from healdit.datasets import ZarrDataset
from healdit.models import HEALVAE
from healdit.train import train
from healdit.utils import load_config
from healdit.utils.parsers import comma_list_to_list, get_train_args


def get_idx(ds: xr.Dataset, date: str) -> int:
    return (ds.valid_time.values >= np.datetime64(date)).argmax()

class ARGS:
    def __init__(self, config_name: str = "healvae_hp", config_path: str = None, overrides: str = "") -> None:
        self.config_name = config_name
        self.config_path = config_path
        self.overrides = overrides

def get_train_args(config_name: str, config_path: str, overrides: str) -> ARGS:
    return ARGS(
        config_name=config_name,
        config_path=config_path,
        overrides=overrides,
    )

if __name__ == "__main__":
    args = get_train_args(
        config_name="healvae_hp_short_wide",
        config_path="/home/benjamin/healdit/config",
        overrides="",
    )

    cfg = load_config(args.config_name, overrides=comma_list_to_list(args.overrides))
    healvae_cfg = instantiate(cfg.healvae)
    train_params = instantiate(cfg.trainparams)
    paths = instantiate(cfg.paths)

    # Train split
    ds = xr.open_zarr(paths.input_data_path, consolidated=True)
    train_start_idx = get_idx(ds, train_params.train_start)
    train_end_idx = get_idx(ds, train_params.train_end)
    val_start_idx = get_idx(ds, train_params.val_start)
    val_end_idx = get_idx(ds, train_params.val_end)
    ds.close()

    train_dataset = ZarrDataset(paths.input_data_path, time_slice=slice(train_start_idx, train_end_idx))
    val_dataset = ZarrDataset(paths.input_data_path, time_slice=slice(val_start_idx, val_end_idx))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_params.batch_size, collate_fn=heal_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_params.batch_size, collate_fn=heal_collate_fn)

    heal_vae = UpDown(healvae_cfg)

    x = next(iter(train_dataloader))
    y = heal_vae(x)
    
    print(x.shape, y.shape)
