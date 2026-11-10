#!/bin/zsh
# Lane 05 arms in the pre-registered order (predictions/context-kv.md addendum). An arm is not started once more than
# 5 h of GPU have passed since the gate smoke (.lab-gpu-start). Each arm is its own results file; nothing is overwritten.
set -u
export HF_HUB_OFFLINE=1
P=~/.pyenv/versions/lewagon/bin/python3
E=mlx-community/gemma-4-e4b-it-4bit; L=LiquidAI/LFM2.5-8B-A1B-MLX-4bit
start=$(date -j -u -f %Y-%m-%dT%H:%M:%SZ "$(cat .lab-gpu-start)" +%s)
arms=(
  "$E ctx-gemma-4-e4b-it-4bit-8k --pad-to 8192"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-8k --pad-to 8192"
  "$E ctx-gemma-4-e4b-it-4bit-2k --pad-to 2048"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-2k --pad-to 2048"
  "$E ctx-gemma-4-e4b-it-4bit-4k --pad-to 4096"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-4k --pad-to 4096"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-8k-kv8 --pad-to 8192 --kv-bits 8"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-8k-kv4 --pad-to 8192 --kv-bits 4"
  "$L ctx-lfm2.5-8b-a1b-mlx-4bit-8k-repeat --pad-to 8192"
  "$E ctx-gemma-4-e4b-it-4bit-8k-repeat --pad-to 8192"
)
for a in "${arms[@]}"; do
  args=(${=a}); name=${args[2]}
  el=$(( $(date -u +%s) - start ))
  if (( el > 18000 )); then echo "$(date -u +%FT%TZ) NOT SAT (5 h gate, ${el}s elapsed): $name"; continue; fi
  echo "$(date -u +%FT%TZ) START $name (${el}s elapsed)"
  $P -m lab context ${args[1]} --name $name --heldout ${args[3,-1]} 2> .lab-scratch/$name.stderr | tail -1
  echo "$(date -u +%FT%TZ) END $name rc=$pipestatus[1]"
done
# Not graded arms: the runtime-exposure checks as train smokes (cheap; after every pre-registered arm).
echo "$(date -u +%FT%TZ) SMOKE e4b kv_bits=8 (expected: abort, NotImplementedError, no results file)"
$P -m lab context $E --name ctx-gate-gemma-4-e4b-it-4bit-8k-kv8-smoke --smoke --pad-to 8192 --kv-bits 8 2> .lab-scratch/e4b-kv8-smoke.stderr; echo "rc=$?"
grep -m1 -o "RuntimeError.*" .lab-scratch/e4b-kv8-smoke.stderr
echo "$(date -u +%FT%TZ) SMOKE LFM max_kv_size=1024 (expected: runs, identical to the 8k smoke)"
$P -m lab context $L --name ctx-gate-lfm2.5-8b-a1b-mlx-4bit-8k-maxkv1024-smoke --smoke --pad-to 8192 --max-kv-size 1024 2>/dev/null | tail -1
echo "$(date -u +%FT%TZ) DONE"
