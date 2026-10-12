#!/bin/bash
# Cron monitor (every 4h): reports trial progress + GPU, scans for OOM/errors, and
# self-heals by restarting the controller if it died with work remaining (but NOT
# after an OOM halt, which needs manual intervention).
RUN=/home/munish-persaud/evo2/local_run
WORK=/home/munish-persaud/evo2/newton
MLOG="$RUN/monitor.log"
CTRL="$RUN/controller.sh"

trial_state(){ # <dir>
  local d=$1
  if [ -f "$d/summary.csv" ]; then echo DONE
  elif [ -d "$d" ]; then echo "partial($(ls "$d" 2>/dev/null | grep -vcE 'json|csv') viruses)"
  else echo pending; fi
}

{
echo "===== MONITOR $(date -Iseconds) ====="
for s in 101 102 103 104 105; do echo "1B s$s: $(trial_state "$WORK/results/local_1b_trial_s$s")"; done
for s in 201 202 203 204 205; do echo "7B s$s: $(trial_state "$WORK/results/local_7b_trial_s$s")"; done

echo "-- running pipeline pids: $(pgrep -f jailbreak_pipeline.py | tr '\n' ' ')"
echo "-- controller alive: $(pgrep -f 'local_run/controller.sh' >/dev/null && echo yes || echo no)"
echo "-- GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"

echo "-- OOM/error scan:"
if grep -liE "out of memory|CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$RUN/logs/"*.log 2>/dev/null; then :; else echo "   none"; fi
echo "-- last status: $(tail -1 "$RUN/status.txt" 2>/dev/null)"

# ---- self-heal ----
if ! pgrep -f 'local_run/controller.sh' >/dev/null 2>&1; then
  remain=0
  # Deliverable is the 5 7B trials; 1B is a single curated trial (s101, already done).
  for s in 201 202 203 204 205; do [ -f "$WORK/results/local_7b_trial_s$s/summary.csv" ] || remain=1; done
  oom_halt=$(tail -1 "$RUN/status.txt" 2>/dev/null | grep -c OOM_HALT)
  if [ "$remain" -eq 1 ] && [ "$oom_halt" -eq 0 ] && ! pgrep -f jailbreak_pipeline.py >/dev/null 2>&1; then
    echo "-- controller dead + work remains: RESTARTING controller"
    setsid nohup bash "$CTRL" >> "$RUN/controller.log" 2>&1 &
  else
    echo "-- controller not running (no restart: remain=$remain oom_halt=$oom_halt pipeline_active=$(pgrep -f jailbreak_pipeline.py >/dev/null && echo yes || echo no))"
  fi
fi
} >> "$MLOG" 2>&1
