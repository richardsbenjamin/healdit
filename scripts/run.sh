#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run.sh <run_name> <config_file> [-v] [-o OVERRIDES]"
    exit 1
fi

RUN="$1"
CONFIG_FILE="$2"

shift 2

VERBOSE_MODE=false
OVERRIDES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|-verbose)
      VERBOSE_MODE=true
      shift # move to next argument
      ;;
    -o|--overrides)
      OVERRIDES="$2"
      shift 2 # move past the flag AND the value
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

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
RUN_CMD="python scripts/${RUN}.py \
    --config-name ${CONFIG_FILE} \
    ${OVERRIDES:+--overrides "$OVERRIDES"}"

# If output file is given, redirect output
if [[ -n "${OUTPUT_FILE}" ]]; then
    ${RUN_CMD} &>> ${OUTPUT_FILE}
    echo "Done. Output saved to ${OUTPUT_FILE}"
else
    ${RUN_CMD}
fi

