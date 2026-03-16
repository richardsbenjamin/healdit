from __future__ import annotations

from hydra.core.config_store import ConfigStore
from healdit.healdit.schemas.config import Config, MSEParams


def register_schemas():
    cs = ConfigStore.instance()

    cs.store(group="modules", name="mse", node=MSEParams)
    cs.store(name="base_config", node=Config)

