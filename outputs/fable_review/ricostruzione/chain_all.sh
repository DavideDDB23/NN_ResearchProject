#!/bin/zsh
# Rerun after the export (restarted at 19:50 on 26 November, after the Mac slept): savings_under_shift.py, M8b on seeds 1 and 2 (results v2), tables and figures of block 3,
# the review scripts of 11-13 November, then pilots B1, B2 and B3 with the commands of the old session (3 workers).
# Can be started again after an interruption: a step whose END is in the log is skipped, and the two
# savings_under_shift.py steps skip the networks already done. Stops at the first failure. Log: outputs/logs/chain.log (same file as stage 1).
cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
export PATH=$HOME/miniforge3/envs/eesavings/bin:$PATH PYTHONDONTWRITEBYTECODE=1
LOG=outputs/logs/chain.log
say() { echo "$(date '+%F %T') $*" >> $LOG }
run() {  # name, then the command; output in outputs/logs/<name>.log
  local name=$1; shift
  grep -q " $name END\$" $LOG && { say "$name already done, skipped"; return }
  say "$name START"
  caffeinate -i "$@" > outputs/logs/$name.log 2>&1 || { say "STOP: $name failed (exit $?)"; exit 1 }
  say "$name END"
}


run savings_under_shift python scripts/5_savings_under_shift/savings_under_shift.py --workers 3

# M8b on seeds 1 and 2, on a copy of the per-network files (README, block 3)
mkdir -p outputs/savings_under_shift_v2/runs && cp -n outputs/savings_under_shift/runs/*.csv outputs/savings_under_shift_v2/runs/
run m8b_seeds12 python scripts/5_savings_under_shift/savings_under_shift.py \
    --results outputs/savings_under_shift_v2/results.csv --only-adaptation --recompute-features --workers 3 \
    --runs outputs/cifar/*/*/seed1 outputs/cifar/*/*/seed2
run savings_figures python scripts/5_savings_under_shift/savings_figures.py \
    --results outputs/savings_under_shift_v2/results.csv --out-dir outputs/block3_review/figures_v2

# review of block 3 (scripts/6_block3_review), the inputs of B1 first
run calibration_budget python scripts/6_block3_review/calibration_budget.py --workers 3
run m2_validity python scripts/6_block3_review/m2_validity.py --workers 3
run adaptation_cost python scripts/6_block3_review/adaptation_cost.py
run decomposition_orders python scripts/6_block3_review/decomposition_orders.py
run intervals python scripts/6_block3_review/intervals.py
run reference_check python scripts/6_block3_review/reference_check.py --workers 3

# pilots of phase B (commands of 14-15 November)
FR=outputs/fable_review/pilots
run b1_levels python $FR/B1_bilancio/b1_levels.py --workers 3
run b1_summary python $FR/B1_bilancio/b1_summary.py
run b2_bn zsh $FR/B2_adattamento/run_b2_bn.sh 0
run b2_heads_c10 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 0 --runs cifar10/joint/seed0 cifar10/joint_sd/seed0
run b2_heads_c100 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 0 --runs cifar100/joint/seed0 cifar100/joint_sd/seed0
run b2_heads_split1 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 1
run b2_summary python $FR/B2_adattamento/b2_summary.py
run b2_mech python $FR/B2_adattamento/b2_mech.py
run b2_mech24 python $FR/B2_adattamento/b2_mech24.py --workers 3
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run b3_decomp python $FR/B3_meccanismo/b3_decomp.py --workers 3
run b3_scores python $FR/B3_meccanismo/b3_scores.py --workers 3
say "stage 2 END"
