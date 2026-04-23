# Getting started (GitHub clone)

This walkthrough assumes an **NVIDIA GPU** for QLoRA training. Four model presets are available (`gpt2`, `qwen3-1.7b`, `phi-3-mini`, `mistral`); **`mistral`** is the default and highest-scoring. **`gpt2`** runs on CPU for small datasets.

## 1. Clone and virtual environment

```bash
git clone <your-repo-url>
cd CS_Fall_Research

python -m venv llm_env
# Windows
llm_env\Scripts\activate.bat
# Linux / macOS
source llm_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-eval.txt
```

Install **CUDA-enabled PyTorch** if `python -c "import torch; print(torch.cuda.is_available())"` prints `False`. See the main README *PyTorch CUDA Support* section.

## 2. Prepare JSONL data

You need one **JSONL** file: one JSON object per line. See [Custom dataset](custom-dataset.md) for the exact schema. For the **general table-to-text workflow** — and how to apply it to your own domain — read [Table to text](table-to-text.md).

Example using the included CSV helper (place your CSVs under `Data/`):

```bash
python scripts/generate_top10_stats_jsonl.py \
  --skaters Data/skaters_24_25.csv \
  --goalies Data/goalies_24_25.csv \
  --output Data/out/my_dataset.jsonl
```

Or bring your own `my_dataset.jsonl` and pass `--data_path`.

## 3. Train (QLoRA)

All four presets use the same entry script. Pick the one that fits your GPU budget:

```bash
# Mistral-7B — best scores, ~5 GB VRAM in 4-bit (default)
python llm_training/player_summary_advanced.py \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_mistral

# Qwen3-1.7B — good quality, ~2 GB VRAM
python llm_training/player_summary_advanced.py \
  --model_preset qwen3-1.7b \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_qwen3

# Phi-3-mini — chat template, ~3 GB VRAM
python llm_training/player_summary_advanced.py \
  --model_preset phi-3-mini \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_phi3

# GPT-2 — smallest, CPU-feasible, useful as baseline
python llm_training/player_summary_advanced.py \
  --model_preset gpt2 \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_gpt2
```

**Train/validation split:** Examples are shuffled before splitting (default seed `42`). Use `--no_shuffle` to keep file order, or `--shuffle_seed N` for a different split.

**Important:** For `--test_only`, pass the same `--model_preset` and `--output_dir` as training so prompts and tokenizer behavior match the saved adapter.

## 4. Evaluate (automatic + efficiency + human rubric)

**A. Export predictions** (validation split):

```bash
python llm_training/player_summary_advanced.py \
  --test_only \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_mistral \
  --num_test -1 \
  --export_predictions outputs/mistral_val_predictions.jsonl
# Change --model_preset and --output_dir for other presets
```

For the **full corpus** (slower):

```bash
python llm_training/player_summary_advanced.py \
  --test_only \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./adapter_mistral \
  --generate_all
# Writes generated_summaries.jsonl: id + generated + timing + peak_gpu_mb
```

**B. Run metrics** (PARENT, BLEU, ROUGE, chrF++, numeric coverage):

```bash
python -m evaluation.run_eval \
  --gold Data/out/my_dataset.jsonl \
  --pred outputs/mistral_val_predictions.jsonl \
  --out outputs/eval_report.json
```

**C. Human ratings** (optional): copy `evaluation/human_rubric_template.csv`, follow `evaluation/RATER_INSTRUCTIONS.txt`, then:

```bash
python -m evaluation.merge_human \
  --human outputs/my_ratings.csv \
  --report outputs/eval_report.json \
  --out outputs/eval_plus_human.json
```

More detail: [Models & evaluation](models-and-evaluation.md).

## Compare all presets

Run training, prediction export, and evaluation for all four models in one shot:

- **Windows:** `.\scripts\run_compare_presets.ps1` (add `-Epochs 1` for a faster trial; `-SkipTrain` if adapters already exist in `outputs/compare_presets/`).
- **Linux / macOS:** `bash scripts/run_compare_presets.sh` (`EPOCHS=1` or `SKIP_TRAIN=1` as needed).

Artifacts land in `outputs/compare_presets/`:
- `eval_report_gpt2.json`, `eval_report_qwen3_17b.json`, `eval_report_phi3_mini.json`, `eval_report_mistral.json`
- `gpt2_val_predictions.jsonl`, `qwen3_17b_val_predictions.jsonl`, `phi3_mini_val_predictions.jsonl`, `mistral_val_predictions.jsonl`
- Adapter configs for each model under `adapter_<preset>/`

Reference results (hockey dataset, 45 val examples) are already committed to `outputs/compare_presets/`. Use `scripts/print_results.py` to print a summary table.
