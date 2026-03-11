from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import hpgeom as hpg
import numpy as np
import torch
from scipy.spatial import transform

from healdit.healdit.utils.geo import (
    get_cartesian_from_spherical,
    get_mesh_lon_lat,
    get_spherical_from_lat_lon_deg,
)
from healdit.healdit.utils.heal import get_neighbours_all

if TYPE_CHECKING:
    from typing import Optional

    from numpy import ndarray
    from torch import Tensor


def _edge_features(relative_pos: ndarray) -> ndarray:
    distances = np.linalg.norm(relative_pos, axis=-1, keepdims=True)
    norm_factor = distances.max()
    
    features = np.concatenate([
        distances / norm_factor,
        relative_pos / norm_factor
    ], axis=-1)
    
    return features

def get_edge_index(
        *,
        nside_in: Optional[int] = None,
        nside_out: Optional[int] = None,
        lon_flat: Optional[ndarray] = None,
        lat_flat: Optional[ndarray] = None,
    ) -> Tensor:
    if nside_in is not None and nside_out is not None:
        if nside_in >= nside_out:
            mesh_in = np.arange(hp.nside2npix(nside_in))
            mesh_out = np.sort(mesh_in % hp.nside2npix(nside_out))
        else:
            mesh_out = np.arange(hp.nside2npix(nside_out))
            mesh_in = np.sort(mesh_out % hp.nside2npix(nside_in))
    elif nside_in is not None and lon_flat is not None:
        mesh_in = np.arange(len(lon_flat))
        mesh_out = hpg.angle_to_pixel(nside_in, lon_flat, lat_flat)
    elif nside_out is not None and lon_flat is not None:
        mesh_in = hpg.angle_to_pixel(nside_out, lon_flat, lat_flat)
        mesh_out = np.arange(len(lon_flat))
       
    return torch.from_numpy(
        np.stack([mesh_in, mesh_out], axis=0)
    )

def get_edge_features(
        edge_index: np.ndarray,
        rec: int | Tuple[np.ndarray],
        send: int | Tuple[np.ndarray],
    ) -> np.ndarray:
    if isinstance(rec, int):
        r_lon_flat, r_lat_flat = get_mesh_lon_lat(rec)
    elif instance(rec, tuple):
        r_lon_flat, r_lat_flat = rec[0], rec[1]
    if isinstance(send, int):
        s_lon_flat, s_lat_flat = get_mesh_lon_lat(send)
    elif isinstance(send, tuple):
        s_lon_flat, s_lat_flat = send[0], send[1]

    senders, receivers = edge_index[0, :], edge_index[1, :]
    
    s_pos, _, _ = get_node_positions(s_lat_flat, s_lon_flat)
    r_pos, r_phi, r_theta = get_node_positions(r_lat_flat, r_lon_flat)

    rot_matrices = get_rotation_matrices(r_phi, r_theta)

    relative_pos = get_relative_space(senders, receivers, s_pos, r_pos, rot_matrices)

    return _edge_features(relative_pos)

def get_mesh_to_mesh_edge_index(nside: int) -> Tensor:
    neighbors = get_neighbours_all(nside=nside)

    target_nodes = np.repeat(np.arange(hp.nside2npix(nside)), 8)
    source_nodes = neighbors.flatten()

    valid_mask = source_nodes != -1
    source_nodes = source_nodes[valid_mask]
    target_nodes = target_nodes[valid_mask]

    edge_index = np.stack([source_nodes, target_nodes], axis=0)
    return torch.from_numpy(edge_index).long()

def get_node_positions(lat: ndarray, lon: ndarray) -> Tuple[ndarray, ...]:
    phi, theta = get_spherical_from_lat_lon_deg(lat, lon)
    pos = np.stack(get_cartesian_from_spherical(phi, theta), axis=-1)
    return pos, phi, theta

def get_relative_space(
        senders: ndarray,
        receivers: ndarray,
        s_pos: ndarray,
        r_pos: ndarray,
        rot_matrices: ndarray,
    ) -> ndarray:
    edge_rot_matrices = rot_matrices[receivers]
    
    r_pos_rotated = rotate_with_matrices(edge_rot_matrices, r_pos[receivers])
    s_pos_rotated = rotate_with_matrices(edge_rot_matrices, s_pos[senders])
    
    return s_pos_rotated - r_pos_rotated

def get_rotation_matrices(phi: ndarray, theta: ndarray) -> ndarray:
    azimuthal_rotation = -phi
    polar_rotation = -theta + np.pi / 2
    
    rot_params = np.stack([azimuthal_rotation, polar_rotation], axis=1)
    rotation_matrices = transform.Rotation.from_euler("zy", rot_params).as_matrix()
    return rotation_matrices

def rotate_with_matrices(
        rotation_matrices: ndarray,
        positions: ndarray,
    ) -> ndarray:
    return np.einsum("...ji,...i->...j", rotation_matrices, positions)

