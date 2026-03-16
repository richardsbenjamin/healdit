#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/healdit"
DATA_DIR="${MODULE_DIR}/data"

START_DATE="2011-01-01"
END_DATE="2016-12-31"
ZARR_INPUT_PATH="gs://healditstorage/era5_data_6h_1degree_1979_2020.zarr"
ZARR_OUTPUT_PATH="${DATA_DIR}/era5_data_6h_1degree_2011_2016.zarr"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH="${MODULE_DIR}"
RUN_CMD="python scripts/bucket2disk.py
    --zarr_input_path ${ZARR_INPUT_PATH}
    --zarr_output_path ${ZARR_OUTPUT_PATH}
    --start-date ${START_DATE}
    --end-date ${END_DATE}"

${RUN_CMD}