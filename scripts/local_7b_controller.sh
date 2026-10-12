#!/bin/bash
# Local (RTX 5090, 32GB) 7B trial controller.
# Paper needs 5 7B trials (camera-ready n>=4); 1B stays a single curated trial
# (already covered by local_1b_trial_s101). Runs all 5 7B trials (seeds 201-205),
# one at a time (32GB card), skipping any already complete. Halts the phase on the
# first OOM so an identical config isn't re-run 4 more times.
set -u

IMG=evo2:pipeline
HF="${HF_CACHE:-$HOME/.cache/huggingface}"
WORK="${EVO2_WORK:-$PWD}"
RUN="${LOCAL_RUN_DIR:-$PWD/local_run}"
LOGDIR="$RUN/logs"
STATUS="$RUN/status.txt"
mkdir -p "$LOGDIR"

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$RUN/controller.log"; }

# Block until no evo2 pipeline is using the GPU.
wait_gpu_free(){
  while pgrep -f "jailbreak_pipeline.py" >/dev/null 2>&1; do
    sleep 60
  done
}

# Trial complete iff its summary.csv exists.
done_check(){ [ -f "$WORK/results/$1/summary.csv" ]; }

# run <model> <seed> <out>. Returns: 0 ok, 42 OOM, other = pipeline rc.
run(){
  local model=$1 seed=$2 out=$3 jlog="$LOGDIR/$3.log"
  wait_gpu_free
  if done_check "$out"; then log "SKIP $out (summary.csv present)"; return 0; fi
  log "START $model seed=$seed -> $out"
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e SKIP_BLAST=1 -e ENTREZ_CACHE_DIR=/work/.entrez_cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$HF":/root/.cache/huggingface -v "$WORK":/work -w /work \
    "$IMG" python3 jailbreak_pipeline.py \
      --model_name "$model" --seed "$seed" --n_samples 1 \
      --output_dir "results/$out" > "$jlog" 2>&1
  local rc=$?
  # Count per-virus OOMs (pipeline catches these and continues to the next virus).
  local oomv; oomv=$(grep -cE "\[FATAL ERROR\].*(out of memory|OutOfMemoryError)" "$jlog" 2>/dev/null)
  if done_check "$out"; then
    # Trial finished (summary.csv written) — possibly partial. Record and CONTINUE.
    log "END $model seed=$seed rc=$rc COMPLETED per-virus-OOM=$oomv -> $out"
    echo "$(date -Iseconds) $out rc=$rc COMPLETED oom_viruses=$oomv" >> "$STATUS"
    return 0
  else
    # No summary.csv = total failure (process died). Halt for intervention.
    log "END $model seed=$seed rc=$rc TOTAL_FAILURE per-virus-OOM=$oomv -> $out"
    echo "$(date -Iseconds) $out rc=$rc TOTAL_FAILURE oom_viruses=$oomv" >> "$STATUS"
    return 42
  fi
}

log "===== CONTROLLER START (pid $$) — 7B x5 ====="

for seed in 201 202 203 204 205; do
  run evo2_7b "$seed" "local_7b_trial_s$seed"
  rc=$?
  if [ "$rc" -eq 42 ]; then
    log "TOTAL_FAILURE_HALT: 7B seed=$seed produced no summary.csv. Halting for manual intervention."
    echo "$(date -Iseconds) OOM_HALT_7B seed=$seed" >> "$STATUS"
    exit 42
  fi
done

log "===== CONTROLLER DONE: all 5 7B trials attempted ====="
echo "$(date -Iseconds) ALL_7B_DONE" >> "$STATUS"
