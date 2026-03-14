from __future__ import annotations

from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace


DATA_DOWNLOAD_PARSER_ARGS = {
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

def get_data_download_args() -> Namespace:
    return get_arg_parser("Download data from EDH.", DATA_DOWNLOAD_PARSER_ARGS)

def get_train_args() -> Namespace:
    return get_arg_parser("Train HealVAE.", TRAIN_PARSER_ARGS)