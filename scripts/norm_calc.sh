#!/bin/bash

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/healdit"
OUTPUT_DIR="${MODULE_DIR}/outputs"

TRAIN_START="1979-01-01T00:00:00"
TRAIN_END="2015-12-31T18:00:00"
ZARR_INPUT_PATH="gs://healditstorage/era5_data_6h_1degree_1979_2020.zarr"
ZARR_OUTPUT_PATH="gs://healditstorage"

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH="${MODULE_DIR}"
RUN_CMD="python scripts/norm_calc.py
    --train_start ${TRAIN_START}
    --train_end ${TRAIN_END}
    --zarr_input_path ${ZARR_INPUT_PATH}
    --zarr_output_path ${ZARR_OUTPUT_PATH}"

${RUN_CMD}