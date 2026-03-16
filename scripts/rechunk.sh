#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/healdit"
DATA_DIR="${MODULE_DIR}/data"

TIME_CHUNK_SIZE=100
LAT_CHUNK_SIZE=-1
LON_CHUNK_SIZE=-1
MAX_MEM="500MB"
ZARR_INPUT_PATH="${DATA_DIR}/era5_data_6h_1degree_2011_2016.zarr"
ZARR_OUTPUT_PATH="${DATA_DIR}/era5_data_6h_1degree_2011_2016_rechunked.zarr"
TEMP_PATH="${DATA_DIR}/era5_data_6h_1degree_2011_2016_temp.zarr"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH="${MODULE_DIR}"
RUN_CMD="python scripts/rechunk.py
    --input-path ${ZARR_INPUT_PATH}
    --output-path ${ZARR_OUTPUT_PATH}
    --temp-path ${TEMP_PATH}
    --time-chunk-size ${TIME_CHUNK_SIZE}
    --lat-chunk-size ${LAT_CHUNK_SIZE}
    --lon-chunk-size ${LON_CHUNK_SIZE}
    --max-mem ${MAX_MEM}"

${RUN_CMD}