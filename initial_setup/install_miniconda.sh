#!/bin/bash

# adapted from: https://waylonwalker.com/install-miniconda/

usage() {
    echo "Usage: $0 [-d <installation_directory>] [-h]"
    echo "  -d <installation_directory>  Specify the base directory for Miniconda installation (default: ~/)."
    echo "  -h                           Display this help message."
}

# Default installation directory
BASE_DIR="${HOME}/"

# Parse command-line options
while getopts ":d:h" opt; do
    case ${opt} in
        d)
            BASE_DIR=$OPTARG
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
    esac
done

mkdir -p ${BASE_DIR}/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${BASE_DIR}/miniconda3/miniconda.sh
bash ${BASE_DIR}/miniconda3/miniconda.sh -b -u -p ${BASE_DIR}/miniconda3
rm -rf ${BASE_DIR}/miniconda3/miniconda.sh
${BASE_DIR}/miniconda3/bin/conda init bash
${BASE_DIR}/miniconda3/bin/conda init zsh