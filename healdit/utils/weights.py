from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn

if TYPE_CHECKING:
    from typing import Any, Tuple

    from healdit.schemas.config import HEALVAEConfig


def apply_posterior_zero(vae: nn.Module, *args: Tuple[Any, ...]) -> None:
    for layer in vae.decoder.layers:
        for block in layer.blocks:
            nn.init.zeros_(block.block.feed_forward2.weight)
            if block.block.feed_forward2.bias is not None:
                nn.init.zeros_(block.block.feed_forward2.bias)

def apply_sqrtn_scale(vae: nn.Module, cfg: HEALVAEConfig) -> None:
    N = sum(cfg.depths)
    for layer in vae.encoder.layers:
        for block in layer.blocks:
            block.feed_forward2.weight.data.mul_(np.sqrt(1 / N))

    for layer in vae.decoder.layers:
        for block in layer.blocks:
            for m in block.z_feedforward.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.mul_(np.sqrt(1 / N))
            block.res_out.feed_forward2.weight.data.mul_(np.sqrt(1 / N))

