#!/bin/bash

set -e

if ! command -v aria2c &> /dev/null; then
  echo "aria2c installation not found."
  exit 1
fi

if ! command -v rsync &> /dev/null; then
  echo "rsync installation not found."
  exit 1
fi

DOWNLOAD_DIR="~/"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
MMCIF_DIR="${ROOT_DIR}/mmcif_files"

echo "Fetching files from rsync.rcsb.org"
mkdir --parents "${RAW_DIR}"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
  "${RAW_DIR}"

echo "Unzip all mmCIF files..."
find "${RAW_DIR}/" -type f -iname "*.gz" -exec gunzip {} +

echo "Flattening all mmCIF files..."
mkdir --parents "${MMCIF_DIR}"
find "${RAW_DIR}" -type d -empty -delete # Delete empty directories.
for subdir in "${RAW_DIR}"/*; do
  mv "${subdir}/"*.cif "${MMCIF_DIR}"
done

echo "Copying files to the cloud"
gsutil cp "${MMCIF_DIR}/*" gs://rcsb_download/
