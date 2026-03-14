from __future__ import annotations

from typing import TYPE_CHECKING

import hpgeom as hpg
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from typing import Tuple

    from numpy import ndarray


def get_cartesian_from_spherical(
        phi: ndarray,
        theta: ndarray,
    ) -> Tuple[ndarray, ...]:
    return (np.cos(phi)*np.sin(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(theta))

def get_lat_lon_flat_grid(lat_res: int, lon_res: int) -> Tuple[ndarray, ndarray]:
    lats = np.linspace(90, -90, lat_res)
    lons = np.linspace(0, 360, lon_res, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.flatten(), lon_grid.flatten() 

def get_mesh_lon_lat(nside: int) -> Tuple[ndarray, ndarray]:
    mesh_lons, mesh_lats = hpg.pixel_to_angle(nside, np.arange(12 * nside**2))
    return mesh_lons, mesh_lats

def get_spherical_from_lat_lon_deg(
        node_lat: ndarray,
        node_lon: ndarray,
    ) -> Tuple[ndarray, ndarray]:
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90 - node_lat)
    return phi, theta

def get_regridded_dataset(ds: xr.Dataset, target_res: float = 1) -> xr.Dataset:
    try:
        import xesmf as xe
    except ImportError:
        raise ImportError(
            "The 'xesmf' package is required for regridding. "
            "Please install it via conda: 'conda install -c conda-forge xesmf'"
        )
    n_lat = int(180 / target_res) + 1 
    n_lon = int(360 / target_res)

    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(-90, 90, n_lat)),
            "lon": (["lon"], np.linspace(0, 360, n_lon, endpoint=False)),
        }
    )
    regridder = xe.Regridder(
        ds, ds_out, method="bilinear", periodic=True
    )
    ds_regridded = regridder(ds)
    return ds_regridded

