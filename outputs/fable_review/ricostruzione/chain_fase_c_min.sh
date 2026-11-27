#!/bin/zsh
# Rerun of phase C limited to the jobs behind the update to the professor (figures 7 and 8 and the numbers of the
# online part): main (M1 M2 M8 M8c) and main_v2 (M4 M7) for the step shift, variants_v2 (boosted audits, continuous
# adaptation, full fallback), the guard of M7wg with its calibration (return to clean and repeated domain changes),
# and their figures. It runs in parallel with chain_all.sh (it needs only the exports), 3 workers. Left for later,
# since the PDF lists them as next steps or does not use them: gamma_clean, the return/sens/weighted/steps/mixed jobs
# of run_c06.sh and phase D. A step whose END is in the log is skipped on a restart, and run.py skips the cases
# already written. Log: outputs/logs/chain_c.log
cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
export PATH=$HOME/miniforge3/envs/eesavings/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG=outputs/logs/chain_c.log
FC=outputs/fable_review/fase_c
say() { echo "$(date '+%F %T') $*" >> $LOG }
run() {  # name, then the command; output in outputs/logs/<name>.log
  local name=$1; shift
  grep -q " $name END\$" $LOG 2>/dev/null && { say "$name already done, skipped"; return }
  say "$name START"
  caffeinate -i "$@" > outputs/logs/$name.log 2>&1 || { say "STOP: $name failed (exit $?)"; exit 1 }
  say "$name END"
}

DOMS35=(); DOMS3=()
for c in gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog \
         brightness contrast elastic_transform pixelate jpeg_compression; do DOMS35+=("$c:3" "$c:5"); DOMS3+=("$c:3"); done
RUNS10=(); RUNS100=()
for r in joint disjoint mixed joint_sd; do for s in 0 1 2; do RUNS10+=("cifar10/$r/seed$s"); RUNS100+=("cifar100/$r/seed$s"); done; done
SEED0_10=(cifar10/joint/seed0 cifar10/disjoint/seed0 cifar10/mixed/seed0 cifar10/joint_sd/seed0)
SEED0_100=(cifar100/joint/seed0 cifar100/disjoint/seed0 cifar100/mixed/seed0 cifar100/joint_sd/seed0)

# C.02 and C.05: the step shift on the 24 networks (figure 7)
run c_main_cifar10 python $FC/run.py --out $FC/main --stream jump --methods M1 M2 M8 M8c --runs $RUNS10 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_v2_cifar10 python $FC/run.py --out $FC/main_v2 --stream jump --methods M4 M7 --runs $RUNS10 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_cifar100 python $FC/run.py --out $FC/main --stream jump --methods M1 M2 M8 M8c --runs $RUNS100 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_main_v2_cifar100 python $FC/run.py --out $FC/main_v2 --stream jump --methods M4 M7 --runs $RUNS100 \
    --domains $DOMS35 --reps 10 --gamma 0.0002 --workers 3
run c_fig_main3 python $FC/figure_timeline.py --dirs $FC/main $FC/main_v2 --methods M1 M2 M4 M7 M8c --severity 3 \
    --out $FC/timeline_main_sev3.png
# C.08: boosted audits, continuous adaptation and full fallback, with the corrected restart
run c_variants_v2_cifar10 python $FC/run.py --out $FC/variants_v2 --stream jump \
    --methods M4 M5 M6 M7 M7f M7c M7t --runs $SEED0_10 --domains $DOMS3 --reps 6 --gamma 0.0002 --workers 3
run c_variants_v2_cifar100 python $FC/run.py --out $FC/variants_v2 --stream jump \
    --methods M4 M5 M7 M7f M7c M7t --runs $SEED0_100 --domains $DOMS3 --reps 4 --gamma 0.0002 --workers 3
run c_fig_phases python $FC/figure_phases.py --dir $FC/variants_v2 --out $FC/phases_sev3.png
# C.21 and C.22: the guard of M7wg, return to clean and repeated domain changes (figure 8)
run c_guard_calibrate python $FC/guard_calibrate.py
C06=$FC/run_c06.sh
export GUARD_KAPPA=0.06
run c06_cifar10_guard zsh $C06 cifar10 6 guard
export GUARD_KAPPA=0.10
run c06_cifar100_guard_return zsh $C06 cifar100 6 guard_return
run c06_cifar100_guard_jm zsh $C06 cifar100 4 guard_jm
run c_fig_guard_return python $FC/figure_timeline.py --dirs $FC/guard_return --methods M4 M7w M7wg --severity 3 \
    --marks 15000 --out $FC/timeline_guard_return.png
run c_fig_guard_mixed python $FC/figure_timeline.py --dirs $FC/guard_mixed --methods M4 M7w M7wg --severity 3 \
    --marks 10000 20000 --out $FC/timeline_guard_mixed.png
say "phase C END"
