from __future__ import annotations

from typing import TYPE_CHECKING

import healpy as hp
import hpgeom as hpg
import numpy as np
import torch
from scipy.spatial import transform

from healdit.utils.geo import (
    get_cartesian_from_spherical,
    get_mesh_lon_lat,
    get_spherical_from_lat_lon_deg,
)
from healdit.utils.heal import get_neighbours_all

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

    from numpy import ndarray
    from torch import Tensor

    from healdit._typing import Location


def _edge_features(relative_pos: ndarray) -> ndarray:
    distances = np.linalg.norm(relative_pos, axis=-1, keepdims=True)
    norm_factor = distances.max()
    
    features = np.concatenate([
        distances / norm_factor,
        relative_pos / norm_factor
    ], axis=-1)
    
    return features

def get_edge_index(
        send: Location,
        rec: Location,
    ) -> Tensor:
    """Get the edge index for a HEALPix grid.
    
    Args:
        send: Nside or (lon, lat) of sending HEALPix grid.
        rec: Nside or (lon, lat) of receiving HEALPix grid.
    
    Returns:
        A tensor containing the edge index.
    """
    if isinstance(send, int) and isinstance(rec, int):
        if send >= rec:
            mesh_in = np.arange(hp.nside2npix(send))
            mesh_out = np.sort(mesh_in % hp.nside2npix(rec))
        else:
            mesh_out = np.arange(hp.nside2npix(rec))
            mesh_in = np.sort(mesh_out % hp.nside2npix(send))

    elif isinstance(send, tuple) and isinstance(rec, int):
        lon, lat = send
        mesh_in = np.arange(len(lon))
        mesh_out = hpg.angle_to_pixel(rec, lon, lat)

    elif isinstance(send, int) and isinstance(rec, tuple):
        lon, lat = rec
        mesh_in = hpg.angle_to_pixel(send, lon, lat)
        mesh_out = np.arange(len(lon))
    else:
        raise ValueError(
            f"Unsupported mapping: {type(send)} to {type(rec)}. "
            "Both cannot be coordinates."
        )

    return torch.from_numpy(np.stack([mesh_in, mesh_out], axis=0))

def get_edge_features(
        edge_index: np.ndarray,
        rec: Location,
        send: Location,
    ) -> np.ndarray:
    """Get the edge features for a HEALPix grid.
    
    Args:
        edge_index: The edge index.
        rec: Nside or (lon, lat) of receiving HEALPix grid.
        send: Nside or (lon, lat) of sending HEALPix grid.
    
    Returns:
        A tensor containing the edge features.

    """
    r_lon, r_lat = resolve_location(rec)
    s_lon, s_lat = resolve_location(send)
    
    s_pos, _, _ = get_node_positions(s_lat, s_lon)
    r_pos, r_phi, r_theta = get_node_positions(r_lat, r_lon)

    rot_matrices = get_rotation_matrices(r_phi, r_theta)

    senders, receivers = edge_index[0, :], edge_index[1, :]

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

def resolve_location(input_val: Location) -> Tuple[ndarray]:
    if isinstance(input_val, int):
        return get_mesh_lon_lat(input_val)
    return input_val

def rotate_with_matrices(
        rotation_matrices: ndarray,
        positions: ndarray,
    ) -> ndarray:
    return np.einsum("...ji,...i->...j", rotation_matrices, positions)

