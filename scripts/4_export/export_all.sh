#!/usr/bin/env bash
# Export the outputs (val, test and CIFAR-C) of every trained network with scripts/4_export/export_outputs.py.
# The CIFAR-C features are saved only for seed 0 (about 8.3 GB per network). Runs already exported are skipped.
# Run it from the repository root, with the eesavings environment active:
#   caffeinate -i bash scripts/4_export/export_all.sh [outputs/cifar]
set -euo pipefail

out_dir="${1:-outputs/cifar}"

for run_dir in "$out_dir"/cifar*/{joint,disjoint,mixed,joint_sd}/seed*; do
  if [[ ! -f "$run_dir/model.pt" || -f "$run_dir/metrics.json" ]]; then
    continue  # not trained yet, or already exported
  fi
  features=""
  if [[ "$run_dir" == */seed0 ]]; then
    features="--features"
  fi
  echo "export $run_dir"
  python scripts/4_export/export_outputs.py --run-dir "$run_dir" $features
done
