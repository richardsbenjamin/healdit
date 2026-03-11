from __future__ import annotations

from hydra.core.config_store import ConfigStore
from healvit.healvit.schemas.config import Config


def register_schemas():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=Config)
