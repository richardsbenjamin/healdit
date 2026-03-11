import torch
from hydra.utils import instantiate

from healdit.healdit import HEALVAE
from healdit.healdit.utils import load_config
from healdit.utils.parsers import get_train_args


if __name__ == "__main__":
    args = get_train_args()
    cfg = load_config(args.config_name, args.overrides)
    healdit_cfg = instantiate(cfg.healvae)

    heal_vae = HEALVAE(healdit_cfg)

    x = torch.rand(1, healdit_cfg.input_feat_dim, *healdit_cfg.lat_lon_res, dtype=torch.float32).reshape(1, healdit_cfg.input_feat_dim, -1)

    output = heal_vae(x)
    print(heal_vae)