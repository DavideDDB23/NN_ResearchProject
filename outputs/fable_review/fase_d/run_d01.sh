#!/bin/zsh
# Phase D, entry D.01 of the notebook: confirmation on the 4 extra corruptions of CIFAR-C (never used before), with
# the methods frozen in Phase C. One run.py at a time, one worker (entry C.19), stop below 30% battery.
#   jump:   the 24 networks, the 4 corruptions at severity 3 and 5, 10 repetitions, M1 M2 M4 M7w M8c
#   return: the 8 networks of seed 0, severity 3, 6 repetitions, M4 M7w M7wg (guard threshold of each dataset)
# Written by Opus 5.5, 15 November 2026. Usage (from the repository root):
#   zsh outputs/fable_review/fase_d/run_d01.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/ddb/miniforge3/envs/eesavings/bin/python
FC=outputs/fable_review/fase_c
FD=outputs/fable_review/fase_d
battery() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%' }

job() {  # the folder name, then the arguments of run.py
  local name=$1; shift
  echo "$(date '+%F %T') $name START battery=$(battery)%" >> $FD/STATUS.txt
  $PY $FC/run.py --out $FD/$name "$@" --workers 1 > $FD/${name}.log 2>&1 &
  local pid=$!
  caffeinate -i -w $pid &
  while kill -0 $pid 2>/dev/null; do
    if [ "$(battery)" -lt 30 ]; then
      echo "$(date '+%F %T') $name STOPPED battery" >> $FD/STATUS.txt
      pkill -f "fase_c/run.py"
      exit 1
    fi
    sleep 30
  done
  wait $pid
  local code=$?
  echo "$(date '+%F %T') $name END exit=$code battery=$(battery)%" >> $FD/STATUS.txt
  cp $FD/$name/args.json $FD/$name/args_$(date '+%H%M%S').json
  [ $code -eq 0 ] || exit $code
}

RUNS=()
for ds in cifar10 cifar100; do for reg in joint disjoint mixed joint_sd; do for s in 0 1 2; do
  RUNS+=("$ds/$reg/seed$s"); done; done; done
EXTRA=(speckle_noise gaussian_blur spatter saturate)
D35=(); D3=()
for c in $EXTRA; do D35+=("$c:3" "$c:5"); D3+=("$c:3"); done

job confirm_jump --stream jump --methods M1 M2 M4 M7w M8c --runs $RUNS --domains $D35 --reps 10 --gamma 0.0002
job confirm_return --stream return --methods M4 M7w M7wg --t0 10000 --t1 15000 --t2 15000 \
    --runs cifar10/joint/seed0 cifar10/disjoint/seed0 cifar10/mixed/seed0 cifar10/joint_sd/seed0 \
    --domains $D3 --reps 6 --gamma 0.0002 --guard-kappa 0.06
job confirm_return --stream return --methods M4 M7w M7wg --t0 10000 --t1 15000 --t2 15000 \
    --runs cifar100/joint/seed0 cifar100/disjoint/seed0 cifar100/mixed/seed0 cifar100/joint_sd/seed0 \
    --domains $D3 --reps 6 --gamma 0.0002 --guard-kappa 0.10
