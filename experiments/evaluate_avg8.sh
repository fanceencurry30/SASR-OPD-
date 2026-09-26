#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MERGED_MODEL GPU_IDS MAX_GENERATION_TOKENS" >&2
  exit 64
fi

model=$(realpath "$1")
gpu_text=$2
max_tokens=$3
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
fire_root=${FIRE_OPD_ROOT:-$release_root/third_party/FiRe-OPD}
data_root=${DATA_ROOT:-$release_root/data}
output_root=${EVAL_OUTPUT_ROOT:-$release_root/outputs/evaluation/$(basename "$model")}
log_root=${LOG_ROOT:-$release_root/logs/evaluation/$(basename "$model")}

[[ "$max_tokens" =~ ^[1-9][0-9]*$ ]] || { echo "invalid token limit" >&2; exit 64; }
test -f "$model/config.json"
test -f "$fire_root/math_eval/eval_math.py"
mkdir -p "$output_root" "$log_root"

IFS=',' read -r -a gpus <<<"$gpu_text"
[[ ${#gpus[@]} -gt 0 ]] || { echo "at least one GPU is required" >&2; exit 64; }
datasets=(aime24 aime25 math500 amc2023 olympiadbench minervamath hmmt25_feb)

run_one() {
  local dataset=$1
  local gpu=$2
  local input=$data_root/eval/$dataset/test.jsonl
  local output=$output_root/$dataset.jsonl
  test -f "$input"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing evaluation output: $output" >&2
    return 73
  fi
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
    export TOKENIZERS_PARALLELISM=false
    cd "$fire_root/math_eval"
    "$python_bin" eval_math.py \
      --input_file "$input" \
      --model_path "$model" \
      --output_file "$output" \
      --max_tokens "$max_tokens" \
      --max_model_len $((max_tokens + 2048)) \
      --temperature 1.0 \
      --top_p 1.0 \
      --max_num_seqs 8 \
      --n 8 \
      --begin_idx -1 \
      --end_idx -1 \
      --seed 42
  ) >"$log_root/$dataset.log" 2>&1
}

pids=()
labels=()
failed=0
wait_wave() {
  local index status
  for index in "${!pids[@]}"; do
    status=0
    wait "${pids[$index]}" || status=$?
    if [[ $status -ne 0 ]]; then
      echo "evaluation failed: ${labels[$index]} (exit $status)" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
}

for index in "${!datasets[@]}"; do
  dataset=${datasets[$index]}
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  run_one "$dataset" "$gpu" &
  pids+=("$!")
  labels+=("$dataset")
  if [[ ${#pids[@]} -eq ${#gpus[@]} ]]; then
    wait_wave
  fi
done
if [[ ${#pids[@]} -gt 0 ]]; then
  wait_wave
fi
[[ $failed -eq 0 ]]

