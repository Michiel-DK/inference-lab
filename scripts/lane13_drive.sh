#!/bin/zsh
# Lane 13 arms in the pre-registered order (predictions/context-kv-truncation.md). Each arm is its own results file.
# Gates before every graded arm (brief, step 2):
#   time   — projected wall = (55 / 3) x the arm's smoke wall x 1.5 (lane 05's rule); not started if it would end after
#            08:00 local ("not sat (time)");
#   memory — the smoke's peak + 2.0 GB (scheduler ram_headroom_gb) must fit free memory, read as vm_stat
#            (free + inactive + speculative) pages x page size ("not sat (memory)");
#   kill (i) — if the maxkv4096 smoke's applied_cache_classes list no RotatingKVCache, arms 1 and 2 are still sat and
#            the repeat is skipped.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export LAB_WILCO_DIR=${LAB_WILCO_DIR:-$HOME/code/Michiel-DK/inference-lab/.claude/worktrees/wilco-pin-62fcd449}
P=~/.pyenv/versions/lewagon/bin/python3
M=mlx-community/Qwen3-4B-4bit
PRED=context-kv-truncation
mkdir -p .lab-scratch
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%Y-%m-%d) 08:00:00" +%s)
(( DEADLINE < $(date +%s) - 43200 )) && DEADLINE=$(( DEADLINE + 86400 ))   # started in the evening: tomorrow's 08:00

log() { echo "$(date -u +%FT%TZ) $*"; }

sit() {   # name, then lab context args
  local name=$1; shift
  log "START $name"
  $P -m lab context $M --name $name --prediction $PRED "$@" 2> .lab-scratch/$name.stderr | tail -3
  log "END $name rc=$pipestatus[1]"
}

free_gb() {
  vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3}
                 END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); printf "%.2f", (f+i+s)*ps/1e9}'
}

smoke_stat() {   # file -> "wall_s peak_gb"
  $P -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d["rows"]
print(sum(x["wall_s"] for x in r), max(c["peak_memory_gb"] for x in r for c in x["calls"]))' "$1"
}

gate() {   # name smoke_file -> 0 to sit
  local name=$1 smoke=$2
  if [[ ! -f $smoke ]]; then log "NOT SAT (no smoke file $smoke): $name"; return 1; fi
  local st=($(smoke_stat $smoke)) fr=$(free_gb)
  local proj=$($P -c "print(int(55/3*${st[1]}*1.5))")
  local end=$(( $(date +%s) + proj ))
  log "GATE $name: smoke wall ${st[1]} s -> projected ${proj} s, ends $(date -r $end +%H:%M) (deadline $(date -r $DEADLINE +%H:%M)); smoke peak ${st[2]} GB + 2.0 vs free ${fr} GB"
  if (( end > DEADLINE )); then log "NOT SAT (time): $name"; return 1; fi
  if $P -c "import sys; sys.exit(0 if ${st[2]} + 2.0 <= ${fr} else 1)"; then return 0; fi
  log "NOT SAT (memory): $name"; return 1
}

S0=results/ctx-gate-qwen3-4b-4bit-8k-smoke.json
S1=results/ctx-gate-qwen3-4b-4bit-8k-maxkv4096-smoke.json
sit ctx-gate-qwen3-4b-4bit-8k-smoke --smoke --pad-to 8192
sit ctx-gate-qwen3-4b-4bit-8k-maxkv4096-smoke --smoke --pad-to 8192 --max-kv-size 4096

kill_i=0
if [[ -f $S1 ]]; then
  cls=$($P -c 'import json,sys; print(json.load(open(sys.argv[1]))["context"]["applied_cache_classes"])' $S1)
  log "smoke maxkv4096 applied_cache_classes: $cls"
  [[ $cls == *RotatingKVCache* ]] || { kill_i=1; log "KILL (i) on the smoke: no RotatingKVCache; arms 1 and 2 still sat, repeat skipped"; }
fi

gate ctx-qwen3-4b-4bit-8k $S0 && sit ctx-qwen3-4b-4bit-8k --heldout --pad-to 8192 --loads 1 --post
gate ctx-qwen3-4b-4bit-8k-maxkv4096 $S1 && sit ctx-qwen3-4b-4bit-8k-maxkv4096 --heldout --pad-to 8192 --max-kv-size 4096 --loads 1 --post
if (( kill_i )); then
  log "NOT SAT (kill (i) fired on the smoke): ctx-qwen3-4b-4bit-8k-maxkv4096-repeat"
else
  gate ctx-qwen3-4b-4bit-8k-maxkv4096-repeat $S1 && sit ctx-qwen3-4b-4bit-8k-maxkv4096-repeat --heldout --pad-to 8192 --max-kv-size 4096 --loads 1 --post
fi
log "DONE"
