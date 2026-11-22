#!/usr/bin/env bash
set -euo pipefail

EHRSQL=${EHRSQL:-EHRSQL}
OPEN_MODELS=("meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "microsoft/phi-4")
API_MODEL=gpt-6-luna
BENCHMARKS=(bird spider ehrsql_mimic_iii ehrsql_eicu)

run() {
  local backend=$1 model=$2 benchmark=$3
  local rewrite=()
  [[ $benchmark == ehrsql_* ]] && rewrite=(--ehrsql-repo "$EHRSQL")
  uv run sqlconsensus run --data "data/${benchmark}.json" --backend "$backend" --model "$model" \
    "${rewrite[@]}" --out-dir "outputs/${benchmark}/${model##*/}"
}

for benchmark in "${BENCHMARKS[@]}"; do
  for model in "${OPEN_MODELS[@]}"; do run vllm "$model" "$benchmark"; done
  run openai "$API_MODEL" "$benchmark"
done
