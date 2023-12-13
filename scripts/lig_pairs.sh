#!/bin/bash

set -e

if ! command -v aria2c &> /dev/null ; then
  echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
  exit 1
fi

if ! command -v gsutil &> /dev/null ; then
  echo "Error: gsutil could not be found. Please install gsutil."
  exit 1
fi

DOWNLOAD_DIR="~"
ROOT_DIR="${DOWNLOAD_DIR}/cluster_info"
SOURCE_URL="https://www.ebi.ac.uk/thornton-srv/databases/pdbsum/data/lig_pairs.lst"

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"

# Copy the Cluster Information to the Cloud.
gsutil cp "${ROOT_DIR}/lig_pairs.lst" "gs://shaker_downloads/lig_pairs.lst"
