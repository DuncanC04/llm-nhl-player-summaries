#!/usr/bin/env bash
# Same pipeline as run_compare_presets.ps1 (train all presets, evaluate each).
# Usage: from repo root: bash scripts/run_compare_presets.sh
# Env: PYTHON=python3  GOLD=Data/out/aiTop10Stats_complete.jsonl  EPOCHS=3  SKIP_TRAIN=1

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
GOLD="${GOLD:-Data/out/aiTop10Stats_complete.jsonl}"
EPOCHS="${EPOCHS:-3}"
NUM_TEST="${NUM_TEST:--1}"

"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print('CUDA OK:', torch.cuda.get_device_name(0))"

if [[ ! -f "$GOLD" ]]; then
  echo "Gold JSONL not found: $GOLD" >&2
  exit 1
fi

OUT="$ROOT/outputs/compare_presets"
mkdir -p "$OUT"

TRAIN="$ROOT/llm_training/player_summary_advanced.py"
PRESETS=("mistral" "phi-3-mini" "gpt2" "qwen3-1.7b")

safe_name() {
  local preset="$1"
  echo "${preset//[-.]/_}"
}

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  for preset in "${PRESETS[@]}"; do
    safe="$(safe_name "$preset")"
    adapter_dir="$OUT/adapter_${safe}"
    pred_path="$OUT/${safe}_val_predictions.jsonl"

    echo "=== Train + export: $preset ==="
    "$PYTHON" "$TRAIN" \
      --data_path "$GOLD" \
      --model_preset "$preset" \
      --output_dir "$adapter_dir" \
      --num_epochs "$EPOCHS" \
      --num_test "$NUM_TEST" \
      --export_predictions "$pred_path"
  done
else
  echo "=== SKIP_TRAIN: test_only + export ==="
  for preset in "${PRESETS[@]}"; do
    safe="$(safe_name "$preset")"
    adapter_dir="$OUT/adapter_${safe}"
    pred_path="$OUT/${safe}_val_predictions.jsonl"
    if [[ ! -d "$adapter_dir" ]]; then
      echo "Missing adapter directory: $adapter_dir" >&2
      exit 1
    fi
    "$PYTHON" "$TRAIN" --test_only --model_preset "$preset" --data_path "$GOLD" --output_dir "$adapter_dir" \
      --num_test "$NUM_TEST" --export_predictions "$pred_path"
  done
fi

for preset in "${PRESETS[@]}"; do
  safe="$(safe_name "$preset")"
  pred_path="$OUT/${safe}_val_predictions.jsonl"
  report_path="$OUT/eval_report_${safe}.json"
  echo "=== evaluation.run_eval: $preset ==="
  "$PYTHON" -m evaluation.run_eval --gold "$GOLD" --pred "$pred_path" --out "$report_path"
done

echo "Done."
for preset in "${PRESETS[@]}"; do
  safe="$(safe_name "$preset")"
  echo "  $preset report: $OUT/eval_report_${safe}.json"
done
