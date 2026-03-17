import logging

import dask
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from edhclient import EDHClient

from healdit.utils import resample_edh_data
from healdit.utils.geo import get_regridded_dataset
from healdit.utils.parsers import get_data2bucket_args, comma_list_to_list

logger = logging.getLogger(__name__)

def get_chunks_spec(time_dim: str, dims: list[str]) -> dict[str, str | int]:
    chunks_spec = {d: -1 for d in dims} 
    chunks_spec[time_dim] = 'auto'
    return chunks_spec

if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=2,          
        threads_per_worker=4,  
        memory_limit='3GB'
    )
    client = Client(cluster)
    
    args = get_data2bucket_args()
    print(f"Downloading data from EDH for the period {args.start_date} to {args.end_date}.")

    edh_client = EDHClient()
    singles_data = edh_client.read_singles()
    pressure_data = edh_client.read_pressure_levels()

    single_level_vars = comma_list_to_list(args.single_level_vars, str)
    pressure_level_vars = comma_list_to_list(args.pressure_level_vars, str)
    pressure_levels = comma_list_to_list(args.pressure_levels, int)
    
    print("Resampling data to target resolution.")
    singles_data = resample_edh_data(singles_data, single_level_vars, args.start_date, args.end_date, args.resample_rate)
    pressure_data = resample_edh_data(pressure_data, pressure_level_vars, args.start_date, args.end_date, args.resample_rate, pressure_levels)

    assert (pressure_data.valid_time != singles_data.valid_time).sum() == 0, "Time indices do not match"

    data_dict = {v: singles_data[v].squeeze(drop=True) for v in single_level_vars}
    for v in pressure_level_vars:
        for lev in pressure_levels:
            data_dict[f"{v}{lev}"] = pressure_data[v].sel(isobaricInhPa=lev).squeeze(drop=True)

    ds = xr.Dataset(data_vars=data_dict)

    with dask.config.set({"array.chunk-size": args.chunk_size}):
        spec = get_chunks_spec(args.time_dim, list(ds.dims))
        ds = ds.squeeze().chunk(spec)

    print("Regridding data to target resolution.")
    regridded = get_regridded_dataset(ds, target_res=args.target_res).squeeze()

    print(f"Saving data to {args.zarr_output_path}.")
    delayed_write = regridded.to_zarr(args.zarr_output_path, mode="w", compute=False)
    future = client.compute(delayed_write)
    progress(future)
    future.result()
    print("Done!")

