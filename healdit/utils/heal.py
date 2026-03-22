from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray


_HEALPY_NEIGHBOUR_NAMES = ['SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'S']
_HEALPY_NEIGHBOUR_IDX   = {name: i for i, name in enumerate(_HEALPY_NEIGHBOUR_NAMES)}

def get_neighbour(neighbours_all: np.ndarray, pixel: int, direction: str) -> int:
    idx = _HEALPY_NEIGHBOUR_IDX[direction]
    return int(neighbours_all[pixel, idx])

def get_neighbours_all(nside: int) -> ndarray:
    npix = hp.nside2npix(nside)
    all_pix = np.arange(npix)
    nbrs = hp.get_all_neighbours(nside, all_pix, nest=True) 
    return nbrs.T

def get_shifted_windows(nbrs_mid: ndarray, npix_window: int) -> Tuple[ndarray, ndarray]:
    CHILD_SW = 0
    CHILD_SE = 1
    CHILD_NW = 2
    CHILD_NE = 3

    all_new_windows = []
    all_masks = []
    seen_windows = set()

    def nbr(pix, direction):
        if pix == -1:
            return -1
        return get_neighbour(nbrs_mid, pix, direction)

    for coarse_pix in range(npix_window):
        base = 4 * coarse_pix

        sw_child = base + CHILD_SW
        se_child = base + CHILD_SE
        nw_child = base + CHILD_NW
        ne_child = base + CHILD_NE

        s_win = [sw_child, nbr(sw_child, 'S'), nbr(sw_child, 'SW'), nbr(sw_child, 'SE')]
        n_win = [nbr(ne_child, 'N'), ne_child, nbr(ne_child, 'NW'), nbr(ne_child, 'NE')]
        w_win = [nbr(nw_child, 'NW'), nbr(nw_child, 'SW'), nbr(nw_child, 'W'), nw_child]
        e_win = [nbr(se_child, 'NE'), nbr(se_child, 'SE'), se_child, nbr(se_child, 'E')]

        for win in [s_win, n_win, w_win, e_win]:
            win_tuple = tuple(win)
            sorted_win = tuple(sorted(win_tuple))
            
            if sorted_win in seen_windows:
                continue
            seen_windows.add(sorted_win)
            
            valid = [p != -1 for p in win]
            mask  = np.array(valid, dtype=bool)
            
            all_new_windows.append(win)
            all_masks.append(mask)

    shifted_windows = np.stack(all_new_windows, axis=0)  
    masks           = np.stack(all_masks, axis=0)

    return shifted_windows, masks