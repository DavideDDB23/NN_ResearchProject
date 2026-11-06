#!/usr/bin/env bash
# Download CIFAR-10-C and CIFAR-100-C (Hendrycks and Dietterich 2019) from Zenodo into data/.
# Each archive is about 2.9 GB and contains 19 corruptions plus labels.npy.
# The md5 values are the ones published on Zenodo (records 2535967 and 3555552).
# Usage: bash scripts/download_cifar_c.sh [cifar10|cifar100|all]
set -euo pipefail

which="${1:-all}"
root="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$root"

declare -a archives=(
  "cifar10 CIFAR-10-C 2535967 56bf5dcef84df0e2308c6dcbcbbd8499"
  "cifar100 CIFAR-100-C 3555552 11f0ed0f1191edbf9fa23466ae6021d3"
)

for entry in "${archives[@]}"; do
  read -r name folder record md5 <<<"$entry"
  if [[ "$which" != "all" && "$which" != "$name" ]]; then
    continue
  fi
  if [[ -s "$root/$folder/labels.npy" ]]; then
    echo "skip $folder (already extracted)"
    continue
  fi
  tar_file="$root/$folder.tar"
  if [[ ! -s "$tar_file" ]]; then
    # download to a .part file first, so an interrupted download is not mistaken for a complete one
    echo "download $folder.tar"
    curl -L --fail --retry 3 -o "$tar_file.part" "https://zenodo.org/records/$record/files/$folder.tar?download=1"
    mv "$tar_file.part" "$tar_file"
  fi
  echo "check md5 of $folder.tar"
  got="$(md5 -q "$tar_file" 2>/dev/null || md5sum "$tar_file" | cut -d' ' -f1)"
  if [[ "$got" != "$md5" ]]; then
    echo "wrong md5 for $folder.tar: $got (expected $md5), deleted: run the script again"
    rm "$tar_file"
    exit 1
  fi
  echo "extract $folder.tar"
  tar -xf "$tar_file" -C "$root"
  rm "$tar_file"
done
