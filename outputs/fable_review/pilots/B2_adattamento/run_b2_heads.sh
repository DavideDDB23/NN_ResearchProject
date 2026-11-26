#!/bin/zsh
# Pilot B2, job B (heads and logit maps re-fitted by distillation, CPU): the 4 networks, 15 corruptions, severity 3
# and 5, splits 0 and 1, with the given number of workers (default 3, one thread each). The c_* configurations (BN
# then heads) need the cache of job A: run job A on split 0 first. The script skips the (network, corruption) pairs
# already done, so it can be run again after an interruption.
# Keeps the Mac awake (caffeinate -i while the job lives); a watchdog checks the battery every 120 s and stops the
# job and its workers below 30%.
# Status lines: outputs/fable_review/pilots/B2_adattamento/STATUS.txt. Log (appended): .../B2_adattamento/b2_heads.log
# Usage (from anywhere): zsh outputs/fable_review/pilots/B2_adattamento/run_b2_heads.sh [workers]

cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
PY=/Users/ddb/miniforge3/envs/eesavings/bin/python
B2=outputs/fable_review/pilots/B2_adattamento
WORKERS=${1:-3}
# what follows the number of workers is passed to b2_heads.py (added by Fable: for example --splits 0 --runs ...)
EXTRA=("${@:2}")
STATUS=$B2/STATUS.txt
LOG=$B2/b2_heads.log
export PYTHONDONTWRITEBYTECODE=1

battery() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%' }

echo "$(date '+%F %T') b2_heads START workers=$WORKERS args=${EXTRA[*]} battery=$(battery)% shell_pid=$$" >> $STATUS
$PY $B2/b2_heads.py --workers $WORKERS $EXTRA >> $LOG 2>&1 &
JOB=$!
caffeinate -i -w $JOB &
echo "$(date '+%F %T') b2_heads python_pid=$JOB" >> $STATUS
while kill -0 $JOB 2>/dev/null; do
  level=$(battery)
  if [ "$level" -lt 30 ]; then
    echo "$(date '+%F %T') b2_heads STOPPED battery=$level% (stop below 30%)" >> $STATUS
    pkill -P $JOB; kill $JOB
    break
  fi
  sleep 120
done
wait $JOB
code=$?
echo "$(date '+%F %T') b2_heads END exit=$code battery=$(battery)%" >> $STATUS
