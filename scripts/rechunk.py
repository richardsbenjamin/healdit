import shutil

import dask
import xarray as xr
import zarr
from rechunker import rechunk

from healdit.utils.parsers import get_rechunk_args


if __name__ == "__main__":
    args = get_rechunk_args()
    dask.config.set(scheduler='synchronous')

    source_data = xr.open_zarr(args.input_path)

    target_chunks = {
        "valid_time": args.time_chunk_size, 
        "lat": args.lat_chunk_size, 
        "lon": args.lon_chunk_size
    }
    rechunk(
        source_data,
        target_chunks=target_chunks,
        max_mem=args.max_mem,
        target_store=args.output_path,
        temp_store=args.temp_path
    ).execute()

    zarr.consolidate_metadata(args.output_path)
    shutil.rmtree(args.temp_path)

    