#!/bin/zsh
# Lane calibrated-4bit arms in the pre-registered order A, B, C (predictions/calibrated-4bit.md). Per arm:
#   gate smoke (--smoke + call-fields support-long-train) -> scripts/lane04_gate.py (time to 05:15 WET, mlx working set)
#   -> memory gate (smoke peak + 2.0 GB <= vm_stat free+inactive+speculative) -> 3 x 83 with --post.
# Swap: re-baselined at each arm's start (the CPU pack build, before any load, had already moved swap 3.04 -> ~4.6 GB);
# a watcher logs vm.swapusage every 5 min and stops the arm if it grows > 2.0 GB above that arm's start ("not sat (swap)").
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export LAB_WILCO_DIR=${LAB_WILCO_DIR:-$HOME/code/Michiel-DK/inference-lab/.claude/worktrees/wilco-pin-62fcd449}
P=~/.pyenv/versions/lewagon/bin/python3
PRED=calibrated-4bit
PK=$HOME/.cache/inference-lab/packs
BASE=dense-gemma-4-e4b-it-4bit
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) 05:15:00" +%s)
mkdir -p .lab-scratch
log() { echo "$(date +%FT%T%z) $*"; }
swap_mb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'; }
free_gb() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3}
  END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%.2f", (f+i+s)*ps/1e9}'; }

arm() {   # tag model
  local tag=$1 model=$2
  local S=calib4-gate-$tag-smoke C=calib4-gate-$tag-call R=calib4-gemma-4-e4b-it-$tag
  log "SMOKE $tag"
  $P -m lab run $model --name $S --prediction $PRED --smoke 2> .lab-scratch/$S.stderr | tail -2
  $P -m lab run $model --name $C --prediction $PRED --exam call-fields --cases support-long-train 2> .lab-scratch/$C.stderr | tail -2
  [[ -f results/$S.json && -f results/$C.json ]] || { log "NOT SAT (smoke did not complete): $tag"; return; }
  local left=$(( (DEADLINE - $(date +%s)) / 60 ))
  $P scripts/lane04_gate.py $S $C $BASE --minutes-left $left || { log "NOT SAT (time or working set): $tag"; return; }
  local peak=$($P -c "import json;print(max(json.load(open(f'results/{r}.json'))['aggregates']['peak_memory_gb_max'] for r in ('$S','$C')))")
  local fr=$(free_gb)
  log "MEMORY $tag: smoke peak $peak + 2.0 vs free $fr GB"
  $P -c "import sys; sys.exit(0 if $peak + 2.0 <= $fr else 1)" || { log "NOT SAT (memory): $tag"; return; }
  local s0=$(swap_mb)
  log "START $R (swap at start ${s0} MB)"
  $P -m lab run $model --name $R --prediction $PRED --exams all --loads 3 --post --scratch .lab-scratch/$R 2> .lab-scratch/$R.stderr &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 300
    local s=$(swap_mb)
    log "SWAP $R ${s} MB (start ${s0}); free $(free_gb) GB"
    if (( s - s0 > 2048 )); then log "NOT SAT (swap): $R, +$(( s - s0 )) MB; stopping"; kill $pid; pkill -P $pid; fi
  done
  wait $pid; log "END $R rc=$?"
}

arm qat-mixed mlx-community/gemma-4-E4B-it-qat-4bit
arm qat-4bit-requant $PK/gemma-4-e4b-it-qat-4bit-requant
arm splice-mlp8 $PK/gemma-4-e4b-it-splice-mlp8
log DONE
