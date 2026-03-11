#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No run name provided."
    echo "Usage: ./run.sh <run_name>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: No config file provided."
    echo "Usage: ./run.sh <run_name> <config_file>"
    exit 1
fi

RUN="$1"
CONFIG_FILE="$2"

if [[ "$3" == "-v" || "$3" == "-verbose" ]]; then
    VERBOSE_MODE=true
fi

# Local paths
HOME_DIR="/home/benjamin"
MODULE_DIR="${HOME_DIR}/healdit"
OUTPUT_DIR="${MODULE_DIR}/outputs"

if [ "$VERBOSE_MODE" = true ]; then
    OUTPUT_FILE=""
    echo "Running in Verbose Mode (printing to console)..."
else
    OUTPUT_FILE="${OUTPUT_DIR}/${RUN}.out"
    if [ -f ${OUTPUT_FILE} ]; then
        rm ${OUTPUT_FILE}
    fi
fi

#############################################################
cd ${MODULE_DIR}
export PYTHONPATH="${MODULE_DIR}"
RUN_CMD="python scripts/${RUN}.py --config-name ${CONFIG_FILE}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
    echo "Done. Output saved to ${OUTPUT_FILE}"
else
    ${RUN_CMD}
fi