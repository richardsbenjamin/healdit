#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/healdit"
OUTPUT_DIR="${MODULE_DIR}/outputs"

CHUNK_SIZE="150MB"
TIME_DIM="valid_time"
TARGET_RES="1.0"
START_DATE="1979-01-01"
END_DATE="1980-12-31"
RESAMPLE_RATE="6h"
PRESSURE_LEVELS="500"
PRESSURE_LEVEL_VARS="z"
SINGLE_LEVEL_VARS="t2m"
ZARR_OUTPUT_PATH="gs://healditstorage/era5_data_6h_1degree_test.zarr"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH="${MODULE_DIR}"
RUN_CMD="python scripts/data_download.py
    --chunk-size ${CHUNK_SIZE}
    --time-dim ${TIME_DIM}
    --target-res ${TARGET_RES}
    --start-date ${START_DATE}
    --end-date ${END_DATE}
    --resample-rate ${RESAMPLE_RATE}
    --pressure-levels ${PRESSURE_LEVELS}
    --pressure-level-vars ${PRESSURE_LEVEL_VARS}
    --single-level-vars ${SINGLE_LEVEL_VARS}
    --zarr_output_path ${ZARR_OUTPUT_PATH}"

${RUN_CMD}