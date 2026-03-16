





if __name__ == "__main__":
    args = get_train_args()
    cfg = load_config(args.config_name, args.overrides)
    healdit_cfg = instantiate(cfg.healvae)

    heal_vae = HEALVAE(healdit_cfg)