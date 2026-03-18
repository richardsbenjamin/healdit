from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import zarr
from torch.utils.data import IterableDataset

from healdit.batch import Batch

if TYPE_CHECKING:
    from typing import Iterator, List, Optional


class ZarrDataset(IterableDataset):

    def __init__(
            self, 
            path: str, 
            variables: List[str] = ["t2m", "z500"], 
            time_slice: Optional[slice] = None,
        ) -> None:
        self.path = path
        self.variables = variables
        self.root = zarr.open_consolidated(path)
        self.chunk_size = self.root[variables[0]].chunks[0]
        self._set_chunk_starts(time_slice)

    def _set_chunk_starts(self, time_slice: slice) -> None:
        time_slice = time_slice if time_slice else slice(0, self.root[self.variables[0]].shape[0])
        self.start_idx = time_slice.start
        self.end_idx = time_slice.stop
        self.chunk_starts = list(range(self.start_idx, self.end_idx, self.chunk_size))

    def __iter__(self) -> Iterator[torch.Tensor]:
        np.random.shuffle(self.chunk_starts)
        for start in self.chunk_starts:
            end = min(start + self.chunk_size, self.end_idx)
            arrays = [self.root[v][start:end] for v in self.variables]
            indices = np.arange(arrays[0].shape[0])
            np.random.shuffle(indices)
            for i in indices:
                yield Batch(data_vars={
                    var_name: torch.from_numpy(arrays[v_idx][i]).reshape(-1, 1).float()
                    for v_idx, var_name in enumerate(self.variables)
                })

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

