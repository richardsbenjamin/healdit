from __future__ import annotations

from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


BUCKET2DISK_PARSER_ARGS = {
    "zarr_input_path": {
        "type": str,
        "required": True,
        "help": "Path to the input zarr file."
    },
    "zarr_output_path": {
        "type": str,
        "required": True,
        "help": "Path to the output zarr file."
    },
    "start-date": {
        "type": str,
        "required": False,
        "default": "1979-01-01",
        "help": "Start date for data retrieval in YYYY-MM-DD format."
    },
    "end-date": {
        "type": str,
        "required": False,
        "default": "2020-12-31",
        "help": "End date for data retrieval in YYYY-MM-DD format."
    },
}

DATA2BUCKET_PARSER_ARGS = {
    "chunk-size": {
        "type": str,
        "required": False,
        "default": "150MB",
        "help": "Target memory size for each dask chunk (e.g., '150MB')."
    },
    "time-dim": {
        "type": str,
        "required": False,
        "default": "valid_time",
        "help": "The name of the time dimension in the source datasets."
    },
    "target-res": {
        "type": float,
        "required": False,
        "default": 1.0,
        "help": "The target horizontal resolution in degrees for regridding."
    },
    "start-date": {
        "type": str,
        "required": True,
        "help": "Start date for data retrieval in YYYY-MM-DD format."
    },
    "end-date": {
        "type": str,
        "required": True,
        "help": "End date for data retrieval in YYYY-MM-DD format."
    },
    "resample-rate": {
        "type": str,
        "required": False,
        "default": "6h",
        "help": "Temporal resampling frequency (e.g., '6h', '1D')."
    },
    "pressure-levels": {
        "type": str,
        "required": False,
        "default": "500",
        "help": "Comma-separated list of pressure levels (hPa) to extract (e.g., '500,850')."
    },
    "pressure-level-vars": {
        "type": str,
        "required": False,
        "default": "z",
        "help": "Comma-separated variables for pressure levels (e.g., 'z,t,u')."
    },
    "single-level-vars": {
        "type": str,
        "required": False,
        "default": "t2m",
        "help": "Comma-separated variables for single levels (e.g., 't2m,tp')."
    },
    "zarr_output_path": {
        "type": str,
        "required": True,
        "help": "Path to the output zarr file."
    }
}

RECHUNK_PARSER_ARGS = {
    "input-path": {
        "type": str,
        "required": True,
        "help": "Path to the input zarr file."
    },
    "output-path": {
        "type": str,
        "required": True,
        "help": "Path to the output zarr file."
    },
    "temp-path": {
        "type": str,
        "required": True,
        "help": "Path to the temporary zarr file."
    },
    "time-chunk-size": {
        "type": int,
        "required": False,
        "default": 1,
        "help": "Size of the time chunk."
    },
    "lat-chunk-size": {
        "type": int,
        "required": False,
        "default": 181,
        "help": "Size of the latitude chunk."
    },
    "lon-chunk-size": {
        "type": int,
        "required": False,
        "default": 360,
        "help": "Size of the longitude chunk."
    },
    "max-mem": {
        "type": str,
        "required": False,
        "default": "2GB",
        "help": "Maximum memory to use for rechunking."
    }
}

TRAIN_PARSER_ARGS = {
    "config-name": {
        "type": str,
        "required": True,
        "help": "Name of the config file to use."
    },
    "overrides": {
        "type": list[str],
        "required": False,
        "default": [],
        "help": "Overrides for the config file."
    },
}


def get_arg_parser(description: str, args_dict: dict[str, dict]) -> Namespace:
    arg_parser = ArgumentParser(description=description)
    for arg, arg_params in args_dict.items():
        arg_parser.add_argument(
            f'--{arg}',
            type=arg_params["type"],
            required=arg_params["required"],
            help=arg_params["help"],
            default=arg_params.get("default"),
        )
    run_args = arg_parser.parse_args()
    return run_args

def get_bucket2disk_args() -> Namespace:
    return get_arg_parser("Write data from bucket to disk.", BUCKET2DISK_PARSER_ARGS)

def get_data2bucket_args() -> Namespace:
    return get_arg_parser("Download data from EDH.", DATA2BUCKET_PARSER_ARGS)

def get_rechunk_args() -> Namespace:
    return get_arg_parser("Rechunk a zarr file.", RECHUNK_PARSER_ARGS)

def get_train_args() -> Namespace:
    return get_arg_parser("Train HealVAE.", TRAIN_PARSER_ARGS)