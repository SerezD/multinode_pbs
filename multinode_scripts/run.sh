#!/bin/bash

echo "activating conda environment: ${ENV_NAME}"
source activate ${ENV_NAME}
NODE_RANK=$OMPI_COMM_WORLD_RANK python ${SCRIPT}

