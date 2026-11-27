#!/bin/zsh
# Remaining steps of the rerun (27 November), in two parallel branches instead of one queue, since phase C is done:
#   A  waits for the M8b job on seeds 1 and 2 already running (pid given as $1), checks that all 16 networks have
#      their M8b rows, then draws the tables and figures of block 3 from results v2;
#   B  the review scripts of block 3 (they read results v1), then pilots B1, B2 and B3 (commands of 14-15 November).
# A step whose END is in outputs/logs/chain.log is skipped, so the script can be started again.
cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
export PATH=$HOME/miniforge3/envs/eesavings/bin:$PATH PYTHONDONTWRITEBYTECODE=1
LOG=outputs/logs/chain.log
M8B_PID=${1:-0}
say() { echo "$(date '+%F %T') $*" >> $LOG }
run() {  # name, then the command; output in outputs/logs/<name>.log
  local name=$1; shift
  grep -q " $name END\$" $LOG && { say "$name already done, skipped"; return 0 }
  say "$name START"
  caffeinate -i "$@" > outputs/logs/$name.log 2>&1 || { say "STOP: $name failed (exit $?)"; return 1 }
  say "$name END"
}

branch_a() {
  while kill -0 $M8B_PID 2>/dev/null; do sleep 60; done
  local n=$(python -c "
import glob, pandas as pd
print(sum(int((pd.read_csv(f, usecols=['method']).method == 'M8b').any()) for f in glob.glob('outputs/savings_under_shift_v2/runs/*.csv')))")
  [ "$n" -eq 24 ] || { say "STOP: m8b_seeds12 ended with $n of 24 networks with M8b rows"; return 1 }
  grep -q " m8b_seeds12 END\$" $LOG || say "m8b_seeds12 END"
  python scripts/5_savings_under_shift/savings_under_shift.py --results outputs/savings_under_shift_v2/results.csv \
      --only-adaptation --workers 1 > outputs/logs/m8b_seeds12_merge.log 2>&1   # nothing left to compute: writes results.csv
  run savings_figures python scripts/5_savings_under_shift/savings_figures.py \
      --results outputs/savings_under_shift_v2/results.csv --out-dir outputs/block3_review/figures_v2
}

branch_b() {
  local FR=outputs/fable_review/pilots
  run calibration_budget python scripts/6_block3_review/calibration_budget.py --workers 3 || return 1
  run m2_validity python scripts/6_block3_review/m2_validity.py --workers 3 || return 1
  run adaptation_cost python scripts/6_block3_review/adaptation_cost.py || return 1
  run decomposition_orders python scripts/6_block3_review/decomposition_orders.py || return 1
  run intervals python scripts/6_block3_review/intervals.py || return 1
  run b1_levels python $FR/B1_bilancio/b1_levels.py --workers 3 || return 1
  run b1_summary python $FR/B1_bilancio/b1_summary.py || return 1
  run b2_bn zsh $FR/B2_adattamento/run_b2_bn.sh 0 || return 1
  run b2_heads_c10 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 0 --runs cifar10/joint/seed0 cifar10/joint_sd/seed0 || return 1
  run b2_heads_c100 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 0 --runs cifar100/joint/seed0 cifar100/joint_sd/seed0 || return 1
  run b2_heads_split1 zsh $FR/B2_adattamento/run_b2_heads.sh 3 --splits 1 || return 1
  run b2_summary python $FR/B2_adattamento/b2_summary.py || return 1
  run b2_mech python $FR/B2_adattamento/b2_mech.py || return 1
  run b2_mech24 python $FR/B2_adattamento/b2_mech24.py --workers 3 || return 1
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 run b3_decomp python $FR/B3_meccanismo/b3_decomp.py --workers 3 || return 1
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 run b3_scores python $FR/B3_meccanismo/b3_scores.py --workers 3 || return 1
}

say "chain_rest START (two branches)"
branch_a & A=$!
branch_b & B=$!
wait $A; ra=$?
wait $B; rb=$?
if [ $ra -eq 0 ] && [ $rb -eq 0 ]; then say "stage 2 END"; else say "STOP: chain_rest ended with A=$ra B=$rb"; fi
