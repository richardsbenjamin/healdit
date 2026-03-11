from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


def get_neighbours_all(nside: int) -> ndarray:
    npix = hp.nside2npix(nside)
    all_pix = np.arange(npix)
    nbrs = hp.get_all_neighbours(nside, all_pix, nest=True) 
    return nbrs.T