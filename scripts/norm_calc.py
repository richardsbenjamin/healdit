import xarray as xr

from healdit.utils.parsers import get_norm_calc_args

if __name__ == "__main__":
    args = get_norm_calc_args()

    data = xr.open_zarr(args.zarr_input_path).sel(valid_time=slice(args.train_start, args.train_end))
    data_mean = data.mean()
    data_std = data.std()

    data_mean.to_zarr(args.zarr_output_path + "/era5_6h_1degree_1979_2015_mean.zarr", mode="w")
    data_std.to_zarr(args.zarr_output_path + "/era5_6h_1degree_1979_2015_std.zarr", mode="w")

