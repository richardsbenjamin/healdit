from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
import torch
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor


class HEALPix:

    def __init__(self, *, n: Optional[int] = None, nside: Optional[int] = None) -> None:
        if n is None and nside is None:
            raise ValueError("Either n or nside must be provided.")
        self.n = n or int(np.log2(nside))
        self.nside = nside or 2 ** n
        self.npix = hp.nside2npix(self.nside)

    def get_edge_index_by_knn(
            self,
            hp_receiver: HEALPix | ndarray,
            n_closest: int = 4,
        ) -> Tensor:
        tree = cKDTree(self.get_vecs())

        if isinstance(hp_receiver, HEALPix):
            vec_receiver = hp_receiver.get_vecs()
            n_receiver_pix = hp_receiver.npix
        else:
            vec_receiver = hp_receiver
            n_receiver_pix = hp_receiver.shape[0]

        _, sender_indices = tree.query(vec_receiver, k=n_closest)
        receiver_indices = np.repeat(np.arange(n_receiver_pix), n_closest)

        return torch.tensor(np.array([sender_indices.flatten(), receiver_indices]))

    def get_vecs(self) -> np.ndarray:
        return np.vstack(hp.pix2vec(self.nside, np.arange(self.npix), nest=True)).T


class HEALWindow(HEALPix):

    def __init__(self, n: int, w: int) -> None:
        # need to put in data validation checks, etc
        super().__init__(n)
        self.w = w
        self.n_sub_w = n - w
        self.nside_window = 2 ** self.n_sub_w
        self.nside_mid = 2 ** (self.n_sub_w + 1)
        self.npix_window = hp.nside2npix(self.nside_window)
        self.pix_per_win = self.npix // self.npix_window
        self.pix_per_mid_win = self.pix_per_win // 4
        self._set_shifted_windows()

    def _set_shifted_windows(self) -> None:
        self.mid_window_nbrs = get_neighbours_all(self.nside_mid)
        self.shifted_windows, self.shifted_windows_mask = get_shifted_windows(self.mid_window_nbrs, self.npix_window)

    def shift_data(self, data: Tensor) -> Tensor:
        fine_indices = np.arange(self.npix)
        flat_idx = self.shifted_windows.ravel() 
        parent_window_indices = fine_indices.reshape(-1, self.pix_per_mid_win)[flat_idx].ravel()
        return (
            data[:, parent_window_indices]
            .reshape(data.shape[0], self.shifted_windows.shape[0], self.pix_per_win, data.shape[-1])
        )

    def unshift_data(self, data: Tensor) -> Tensor:
        flat_win = self.shifted_windows.ravel()
        sort_idx = np.argsort(flat_win)
        valid_sort_idx = sort_idx[flat_win[sort_idx] != -1]
        data = data.reshape(data.shape[0], -1, data.shape[-1])
        return data[:, valid_sort_idx].reshape(data.shape[0], -1, data.shape[-1])

