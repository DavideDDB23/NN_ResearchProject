#!/bin/zsh
# Pilot B2, job A (BatchNorm statistics on the new domain, one process on MPS): the 4 networks, 15 corruptions,
# severity 3 and 5, the splits given (default 0 1). The script skips the (network, corruption) pairs already done for
# those splits, so it can be run again after an interruption; a run with split 1 after one with split 0 adds rows.
# Keeps the Mac awake (caffeinate -i while the job lives); a watchdog checks the battery every 120 s and stops the
# job below 30%.
# Status lines: outputs/fable_review/pilots/B2_adattamento/STATUS.txt. Log (appended): .../B2_adattamento/b2_bn.log
# Usage (from anywhere): zsh outputs/fable_review/pilots/B2_adattamento/run_b2_bn.sh [splits...]

cd /Users/ddb/Desktop/NN_ResearchProject || exit 1
PY=/Users/ddb/miniforge3/envs/eesavings/bin/python
B2=outputs/fable_review/pilots/B2_adattamento
STATUS=$B2/STATUS.txt
LOG=$B2/b2_bn.log
SPLITS=("$@")
if (( ${#SPLITS} == 0 )); then SPLITS=(0 1); fi
export PYTHONDONTWRITEBYTECODE=1

battery() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%' }

echo "$(date '+%F %T') b2_bn START splits=${SPLITS[*]} battery=$(battery)% shell_pid=$$" >> $STATUS
$PY $B2/b2_bn.py --splits $SPLITS >> $LOG 2>&1 &
JOB=$!
caffeinate -i -w $JOB &
echo "$(date '+%F %T') b2_bn python_pid=$JOB" >> $STATUS
while kill -0 $JOB 2>/dev/null; do
  level=$(battery)
  if [ "$level" -lt 30 ]; then
    echo "$(date '+%F %T') b2_bn STOPPED battery=$level% (stop below 30%)" >> $STATUS
    pkill -P $JOB; kill $JOB
    break
  fi
  sleep 120
done
wait $JOB
code=$?
echo "$(date '+%F %T') b2_bn END exit=$code battery=$(battery)%" >> $STATUS
