#!/usr/bin/env bash
# Download the precomputed logits released with Fast yet Safe (metodj/RC-EENN, img_cls/README.md)
# into data/fys/<model>/<dataset>.p. ImageNet files are about 3.4 GB in total, CIFAR100 about 60 MB.
# Usage: bash scripts/1_data/download_fys_logits.sh [imagenet|cifar100|all]
set -euo pipefail

which="${1:-all}"
root="$(cd "$(dirname "$0")/../.." && pwd)/data/fys"

declare -a files=(
  "1FZhXaSigCbzt6IzfSIGitKef0d1WehJO msdnet imagenet"
  "1ZWJZDOxRMUmj5EC04i1cJMJsh2Vc17Pt dvit imagenet"
  "16Az4vqekIIar5yfkdS_0Euo7LqlCjesH dynperc imagenet"
  "119xDvSDx-cHDH70qsgGczLDLSDrKwCHU l2wden imagenet"
  "1EA92VVaHO_5dQEwjcKpbpLsqbRdyUNFs msdnet cifar100"
  "1phOjaDl0vl7qkVukS0Soyct0BFkrKO3K dvit cifar100"
  "1VqQYVu67q9ikZ1L9oNFetQhym6ofTAcM l2wden cifar100"
)

for entry in "${files[@]}"; do
  read -r id model dataset <<<"$entry"
  if [[ "$which" != "all" && "$which" != "$dataset" ]]; then
    continue
  fi
  out="$root/$model/$dataset.p"
  if [[ -s "$out" ]]; then
    echo "skip $out (already present)"
    continue
  fi
  mkdir -p "$root/$model"
  echo "download $model/$dataset"
  gdown --quiet "$id" -O "$out"
done
