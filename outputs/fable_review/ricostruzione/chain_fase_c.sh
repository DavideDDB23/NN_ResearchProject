#!/bin/zsh
# Rerun of phase C (online streams, 15 November) after chain_all.sh, with the final code recovered on 26 November and
# 3 workers. Only the final version of every job: main (M1 M2 M8 M8c) and main_v2 (M4 M7 with the corrected restart of
# lambda) as in figure 7 of the update to the professor; variants_v2; the jobs of run_c06.sh in the order of the old
# session; guard_calibrate.py; the guard jobs (figure 8); the figures. The superseded folders (variants, return of
# 13:05, the first M4/M7 of main) are not rerun. A step whose END is in the log is skipped on a restart, and run.py
# skips the cases already written. Log: outputs/logs/chain.log
cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
export PATH=$HOME/miniforge3/envs/eesavings/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG=outputs/logs/chain.log
FC=outputs/fable_review/fase_c
say() { echo "$(date '+%F %T') $*" >> $LOG }
run() {  # name, then the command; output in outputs/logs/<name>.log
  local name=$1; shift
  grep -q " $name END\$" $LOG && { say "$name already done, skipped"; return }
  say "$name START"
  caffeinate -i "$@" > outputs/logs/$name.log 2>&1 || { say "STOP: $name failed (exit $?)"; exit 1 }
  say "$name END"
}

until grep -q " stage 2 END\$" $LOG; do
  grep -q "STOP:" $LOG && { say "phase C not started: an earlier step stopped"; exit 1 }
  sleep 60
done

DOMS35=(); DOMS3=()
for c in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog \
         brightness contrast elastic_transform pixelate jpeg_compression; do DOMS35+=("$c:3" "$c:5"); DOMS3+=("$c:3"); done
RUNS10=(); RUNS100=()
for r in joint disjoint mixed joint_sd; do for s in 0 1 2; do RUNS10+=("cifar10/$r/seed$s"); RUNS100+=("cifar100/$r/seed$s"); done; done
SEED0_10=(cifar10/joint/seed0 cifar10/disjoint/seed0 cifar10/mixed/seed0 cifar10/joint_sd/seed0)
SEED0_100=(cifar100/joint/seed0 cifar100/disjoint/seed0 cifar100/mixed/seed0 cifar100/joint_sd/seed0)

# C.01: the choice of gamma on clean streams
run c_gamma_clean python $FC/run.py --out $FC/gamma_clean --stream clean \
    --methods M1 M4@0.0001 M4@0.0002 M4@0.0005 M4@0.001 M4@0.002 --runs $SEED0_10 $SEED0_100 --reps 10 --workers 3
# C.02 and C.05: the main job (figure 7)
run c_main_cifar10 python $FC/run.py --out $FC/main --stream jump --methods M1 M2 M8 M8c --runs $RUNS10 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_cifar100 python $FC/run.py --out $FC/main --stream jump --methods M1 M2 M8 M8c --runs $RUNS100 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_v2_cifar10 python $FC/run.py --out $FC/main_v2 --stream jump --methods M4 M7 --runs $RUNS10 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_v2_cifar100 python $FC/run.py --out $FC/main_v2 --stream jump --methods M4 M7 --runs $RUNS100 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
# C.08: the variants with the corrected restart
run c_variants_v2_cifar10 python $FC/run.py --out $FC/variants_v2 --stream jump \
    --methods M4 M5 M6 M7 M7f M7c M7t --runs $SEED0_10 --domains $DOMS3 --reps 6 --gamma 0.0002 --workers 3
run c_variants_v2_cifar100 python $FC/run.py --out $FC/variants_v2 --stream jump \
    --methods M4 M5 M7 M7f M7c M7t --runs $SEED0_100 --domains $DOMS3 --reps 4 --gamma 0.0002 --workers 3
# C.06 to C.20: return to clean, sensitivity, weighted fit, steps and mixed corruptions (run_c06.sh)
C06=$FC/run_c06.sh
run c06_cifar10_all zsh $C06 cifar10 6 all
run c06_cifar10_weighted zsh $C06 cifar10 6 weighted
run c06_cifar10_segments zsh $C06 cifar10 6 segments
run c06_cifar10_w zsh $C06 cifar10 6 w
run c06_cifar100_weighted zsh $C06 cifar100 4 weighted
run c06_cifar100_return zsh $C06 cifar100 6 return
run c06_cifar100_w_return zsh $C06 cifar100 6 w_return
run c06_cifar100_segments zsh $C06 cifar100 4 segments
run c06_cifar100_sens zsh $C06 cifar100 6 sens
# C.21 and C.22: the guard of M7wg (figure 8)
run c_guard_calibrate python $FC/guard_calibrate.py
export GUARD_KAPPA=0.06
run c06_cifar10_guard zsh $C06 cifar10 6 guard
export GUARD_KAPPA=0.10
run c06_cifar100_guard_return zsh $C06 cifar100 6 guard_return
run c06_cifar100_guard_jm zsh $C06 cifar100 4 guard_jm
# figures, with the commands of 15 November
TL=$FC/figure_timeline.py
run c_fig_main3 python $TL --dirs $FC/main $FC/main_v2 --methods M1 M2 M4 M7 M8c --severity 3 --out $FC/timeline_main_sev3.png
run c_fig_main5 python $TL --dirs $FC/main $FC/main_v2 --methods M1 M2 M4 M7 M8c --severity 5 --out $FC/timeline_main_sev5.png
run c_fig_return python $TL --dirs $FC/return $FC/return_w --methods M4 M7 M7c M7cw M8c --severity 3 --marks 15000 \
    --out $FC/timeline_return_sev3.png
run c_fig_steps python $TL --dirs $FC/steps $FC/steps_w --methods M2 M4 M7 M7t M7c M7cw --severity 3 \
    --marks 6000 12000 18000 24000 --out $FC/timeline_steps.png
run c_fig_mixed python $TL --dirs $FC/mixed $FC/mixed_w --methods M2 M4 M7 M7t M7c M7cw --severity 3 \
    --marks 10000 20000 --out $FC/timeline_mixed.png
run c_fig_guard_return python $TL --dirs $FC/guard_return --methods M4 M7w M7wg --severity 3 --marks 15000 \
    --out $FC/timeline_guard_return.png
run c_fig_guard_mixed python $TL --dirs $FC/guard_mixed --methods M4 M7w M7wg --severity 3 --marks 10000 20000 \
    --out $FC/timeline_guard_mixed.png
run c_fig_phases python $FC/figure_phases.py --dir $FC/variants_v2 --out $FC/phases_sev3.png
say "phase C END"
