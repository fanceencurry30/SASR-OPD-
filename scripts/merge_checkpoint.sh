#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 UPSTREAM_REPOSITORY ACTOR_CHECKPOINT OUTPUT_MODEL" >&2
  exit 64
fi

upstream=$(realpath "$1")
actor=$(realpath "$2")
output=$3
python_bin=${PYTHON:-python}

test -d "$upstream/verl/verl"
test -d "$actor"
mkdir -p "$(dirname "$output")"

cd "$upstream/verl"
exec "$python_bin" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$actor" \
  --target_dir "$output"

