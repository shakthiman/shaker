#!/bin/bash

set -e

if ! command -v aria2c &> /dev/null ; then
  echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
  exit 1
fi

DOWNLOAD_DIR="~"
ROOT_DIR="${DOWNLOAD_DIR}/cluster_info"
SOURCE_URL="https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt"

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"

# Copy the Cluster Information to the Cloud.
gsutil cp "${ROOT_DIR}/clusters-by-entity-40.txt" "gs://clustered_pdb/clusters-by-entity-40.txt"
