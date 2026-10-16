#!/bin/bash
# cpu_sample.sh <seconds>
#
# Prints the average CPU utilisation over <seconds> from /proc/stat:
#   busy%  user%  sys%  iowait%  steal%  and the number of runnable threads.
# Used to tell "CPU-bound" apart from "contention-bound": if the insert run
# does not saturate the box, the limit is lock/page contention, and splitting
# the write path across more regions (multi-hot) can recover it.
set -u
SECS=${1:-10}

read -r _ u n s i w irq sirq st _ < <(grep '^cpu ' /proc/stat)

u0=$u; n0=$n; s0=$s; i0=$i; w0=$w; irq0=$irq; sirq0=$sirq; st0=$st
tot0=$((u+n+s+i+w+irq+sirq+st))

# Sample runnable+uninterruptible threads (procs_running / procs_blocked).
peak_run=0; sum_run=0; nsm=0
end=$(( $(date +%s) + SECS ))
while [ "$(date +%s)" -lt "$end" ]; do
    r=$(awk '/^procs_running/{print $2}' /proc/stat)
    sum_run=$((sum_run + r)); nsm=$((nsm + 1))
    [ "$r" -gt "$peak_run" ] && peak_run=$r
    sleep 1
done

read -r _ u n s i w irq sirq st _ < <(grep '^cpu ' /proc/stat)
tot1=$((u+n+s+i+w+irq+sirq+st))
DT=$((tot1 - tot0))
[ "$DT" -le 0 ] && DT=1
pct() { awk -v d="$DT" -v v="$1" 'BEGIN{printf "%.1f", 100*v/d}'; }
echo "cpu busy=$(pct $(( (u-u0)+(n-n0)+(s-s0)+(irq-irq0)+(sirq-sirq0) )))% user=$(pct $((u-u0)))% sys=$(pct $((n0*0+n-n0)))% iowait=$(pct $((w-w0)))% steal=$(pct $((st-st0)))% procs_running_avg=$(awk -v s="$sum_run" -v n="$nsm" 'BEGIN{printf "%.1f", s/n}') peak=$peak_run"
