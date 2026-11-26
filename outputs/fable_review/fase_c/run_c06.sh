#!/bin/zsh
# Phase C, entry C.06 of the notebook: the return to the clean domain and the sensitivity to pi and gamma, on the 4
# networks of seed 0 of one dataset, 15 corruptions at severity 3. One run.py at a time with 3 workers; it stops every
# job below 30% battery. Written by Opus 5.5, 15 November 2026.
# Usage (from the repository root):
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar10 6 all       (return, then the three sensitivity folders)
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar100 6 return
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar100 6 sens
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar10 6 segments   (entry C.07: steps of severity, mixed corruptions)
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar10 6 weighted   (entry C.09: the weighted fit, M7w and M7cw)
#   zsh outputs/fable_review/fase_c/run_c06.sh cifar10 6 w          (entry C.13: M7cw on return, steps, mixed)
set -u
DS=$1; REPS=$2; WHAT=$3
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/ddb/miniforge3/envs/eesavings/bin/python
FC=outputs/fable_review/fase_c
DOMS=()
for c in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog \
         brightness contrast elastic_transform pixelate jpeg_compression; do DOMS+=("$c:3"); done
RUNS=($DS/joint/seed0 $DS/disjoint/seed0 $DS/mixed/seed0 $DS/joint_sd/seed0)
battery() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%' }

job() {  # the folder name, then the arguments of run.py that change between jobs
  local name=$1; shift
  echo "$(date '+%F %T') $name $DS START (reps $REPS) battery=$(battery)%" >> $FC/STATUS.txt
  # WORKERS=2 in the environment when the charger does not keep up with three (CIFAR100, 15 November afternoon)
  $PY $FC/run.py --out $FC/$name "$@" --runs $RUNS --domains $DOMS --reps $REPS --workers ${WORKERS:-3} \
      > $FC/${name}_$DS.log 2>&1 &
  local pid=$!
  caffeinate -i -w $pid &
  while kill -0 $pid 2>/dev/null; do
    if [ "$(battery)" -lt 30 ]; then
      echo "$(date '+%F %T') $name $DS STOPPED battery" >> $FC/STATUS.txt
      pkill -f "fase_c/run.py"
      exit 1
    fi
    sleep 30
  done
  wait $pid
  local code=$?
  echo "$(date '+%F %T') $name $DS END exit=$code battery=$(battery)%" >> $FC/STATUS.txt
  cp $FC/$name/args.json $FC/$name/args_$DS.json  # run.py writes args.json at every launch: keep one per dataset
  [ $code -eq 0 ] || exit $code
}

if [[ $WHAT == all || $WHAT == return ]]; then
  job return --stream return --methods M4 M7 M7c M8c --t0 10000 --t1 15000 --t2 15000 --gamma 0.0002
fi
if [[ $WHAT == all || $WHAT == sens ]]; then
  job sens_pi002 --stream jump --methods M4 M7 --pi 0.02 --gamma 0.0002
  job sens_pi01 --stream jump --methods M4 M7 --pi 0.1 --gamma 0.0002
  job sens_gamma --stream jump --methods M4@0.0001 M4@0.0005 M4@0.001 M7@0.0005 --gamma 0.0002
fi
if [[ $WHAT == weighted ]]; then  # entry C.09: the weighted fit of the maps, with M4 and M7 as a check against
  # variants_v2 (same repetitions: 6 on CIFAR10, 4 on CIFAR100)
  job variants_w --stream jump --methods M4 M7 M7w M7cw --gamma 0.0002
fi
# entry C.13: M7cw on the return, the steps and the mixed corruptions, with M4 as the paired reference (separate
# folders: run.py skips the cases already written, so no method is added to the old folders); w = both parts
if [[ $WHAT == w || $WHAT == w_return ]]; then
  job return_w --stream return --methods M4 M7cw --t0 10000 --t1 15000 --t2 15000 --gamma 0.0002
fi
if [[ $WHAT == w || $WHAT == w_segments ]]; then
  job steps_w --stream steps --methods M4 M7cw --gamma 0.0002
  job mixed_w --stream mixed --methods M4 M7cw --gamma 0.0002
fi
# entry C.21: the guard of M7wg; GUARD_KAPPA from guard_calibrate.py (0.06 on CIFAR10, 0.10 on CIFAR100)
if [[ $WHAT == guard || $WHAT == guard_return ]]; then
  job guard_return --stream return --methods M4 M7w M7wg --t0 10000 --t1 15000 --t2 15000 --gamma 0.0002 \
      --guard-kappa $GUARD_KAPPA
fi
if [[ $WHAT == guard || $WHAT == guard_jm ]]; then
  job guard_jump --stream jump --methods M4 M7wg --gamma 0.0002 --guard-kappa $GUARD_KAPPA
  job guard_mixed --stream mixed --methods M4 M7w M7wg --gamma 0.0002 --guard-kappa $GUARD_KAPPA
fi
if [[ $WHAT == segments ]]; then  # entry C.07: steps of severity and mixed corruptions
  job steps --stream steps --methods M2 M4 M7 M7t M7c --gamma 0.0002
  job mixed --stream mixed --methods M2 M4 M7 M7t M7c --gamma 0.0002
fi
