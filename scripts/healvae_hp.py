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

if __name__ == "__main__":
    args = get_train_args()

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

    heal_vae = HEALVAE(healvae_cfg)
    heal_vae = heal_vae.to(train_params.device)

    if train_params.weight_init is not None:
        train_params.weight_init(heal_vae, healvae_cfg)

    train_history = train(
        model=heal_vae,
        loader=train_dataloader,
        params=train_params,
    )

    with open(paths.history_path, "wb") as f:
        pickle.dump(train_history, f)


