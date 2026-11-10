#!/bin/zsh
# Lane 06 arms in the pre-registered order (predictions/speculative.md addendum). Each arm is its own results file.
set -u
export HF_HUB_OFFLINE=1
P=~/.pyenv/versions/lewagon/bin/python3
T=mlx-community/gemma-4-e4b-it-4bit; D=mlx-community/gemma-4-e2b-it-4bit
arms=(
  "spec-gate-gemma-4-e4b-it-4bit-control-smoke --smoke"
  "spec-gate-gemma-4-e4b-it-4bit-e2b-k3-smoke --smoke --draft $D --num-draft-tokens 3"
  "spec-gemma-4-e4b-it-4bit-control --all"
  "spec-gemma-4-e4b-it-4bit-e2b-k3 --all --draft $D --num-draft-tokens 3"
  "spec-gemma-4-e4b-it-4bit-e2b-k3-repeat --all --draft $D --num-draft-tokens 3"
)
for a in "${arms[@]}"; do
  args=(${=a}); name=${args[1]}
  echo "$(date -u +%FT%TZ) START $name"
  $P -m lab spec $T --name $name ${args[2,-1]} 2> .lab-scratch/$name.stderr | tail -1
  echo "$(date -u +%FT%TZ) END $name rc=$pipestatus[1]"
done
echo "$(date -u +%FT%TZ) DONE"
