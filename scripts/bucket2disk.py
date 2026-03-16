import xarray as xr

from healdit.utils.parsers import get_bucket2disk_args


if __name__ == "__main__":
    args = get_bucket2disk_args()
    
    data = xr.open_zarr(args.zarr_input_path)
    data = data.sel(valid_time=slice(args.start_date, args.end_date))
    data.to_zarr(args.zarr_output_path, mode="w")