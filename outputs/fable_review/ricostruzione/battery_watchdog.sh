#!/bin/zsh
# Battery watchdog for the rerun jobs: every 60 s it reads the battery; below LOW it pauses (SIGSTOP) every Python
# process of the eesavings environment, above HIGH it resumes them (SIGCONT). Pausing instead of killing keeps the
# jobs where they are while the 30 W charger refills the battery. Exits when outputs/logs/watchdog.stop exists.
LOW=${LOW:-25}; HIGH=${HIGH:-45}
LOG=/Users/ddb/Desktop/NN_ResearchProject/outputs/logs/watchdog.log
STOPFILE=/Users/ddb/Desktop/NN_ResearchProject/outputs/logs/watchdog.stop
PATTERN="miniforge3/envs/eesavings/bin/python"
paused=0
battery() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%' }
echo "$(date '+%F %T') watchdog START low=$LOW high=$HIGH battery=$(battery)%" >> $LOG
while [ ! -f $STOPFILE ]; do
  b=$(battery)
  if [ $paused -eq 0 ] && [ "$b" -lt $LOW ]; then
    pkill -STOP -f "$PATTERN" && echo "$(date '+%F %T') PAUSED jobs at battery=$b%" >> $LOG
    paused=1
  elif [ $paused -eq 1 ] && [ "$b" -ge $HIGH ]; then
    pkill -CONT -f "$PATTERN"; echo "$(date '+%F %T') RESUMED jobs at battery=$b%" >> $LOG
    paused=0
  fi
  sleep 60
done
[ $paused -eq 1 ] && pkill -CONT -f "$PATTERN"
echo "$(date '+%F %T') watchdog END battery=$(battery)%" >> $LOG
