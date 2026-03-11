import torch
from hydra.utils import instantiate

from healvit.healvit import HEALVAE
from healvit.healvit.utils import load_config
from healvit.utils.parsers import get_train_args


if __name__ == "__main__":
    args = get_train_args()
    cfg = load_config(args.config_name, args.overrides)
    healvit_cfg = instantiate(cfg.healvae)

    heal_vae = HEALVAE(healvit_cfg)

    x = torch.rand(1, healvit_cfg.input_feat_dim, *healvit_cfg.lat_lon_res, dtype=torch.float32).reshape(1, healvit_cfg.input_feat_dim, -1)

    output = heal_vae(x)
    print(heal_vae)