# Getting started (GitHub clone)

This walkthrough assumes an **NVIDIA GPU** for QLoRA training. The default preset is **Mistral-7B** (~8GB+ VRAM with 4-bit); **`--model_preset phi-3-mini`** is a smaller option on the same pipeline.

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

You need one **JSONL** file: one JSON object per line. See [Custom dataset](custom-dataset.md) for the exact schema. For the **general table-to-text workflow** and how this repo wires prompts, training, and PARENT evaluation together, read [Table to text](table-to-text.md).

Example using the included CSV helper (place your CSVs under `Data/`):

```bash
python scripts/generate_top10_stats_jsonl.py \
  --skaters Data/skaters_24_25.csv \
  --goalies Data/goalies_24_25.csv \
  --output Data/out/my_dataset.jsonl
```

Or bring your own `my_dataset.jsonl` and pass `--data_path`.

## 3. Train (QLoRA)

Default preset is **Mistral-7B**. Training and inference code lives in **`llm_training/player_summary/`**; the script you run is still **`llm_training/player_summary_advanced.py`**.

```bash
python llm_training/player_summary_advanced.py \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./player_summary_model \
  --num_epochs 3
```

**Phi-3-mini** (chat template for SFT and generation):

```bash
python llm_training/player_summary_advanced.py \
  --model_preset phi-3-mini \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./player_summary_phi3 \
  --num_epochs 3
```

**Train/validation split:** Examples are **shuffled** before splitting (default seed `42`) so validation is not tied to JSONL row order. Use **`--no_shuffle`** to keep file order, or **`--shuffle_seed N`** to reproduce a different split.

**Important:** For **`--test_only`**, pass the same **`--model_preset`** (and usually the same **`--output_dir`**) as training so prompts and tokenizer behavior match the saved adapter.

## 4. Evaluate (automatic + efficiency + human rubric)

**A. Export predictions** (validation split, same `id`s as gold):

```bash
python llm_training/player_summary_advanced.py \
  --test_only \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./player_summary_model \
  --num_test -1 \
  --export_predictions outputs/mistral_val_predictions.jsonl
```

If you trained with **`--model_preset phi-3-mini`**, add **`--model_preset phi-3-mini`** and point **`--output_dir`** at that run’s adapter directory.

For **all rows** (full corpus metrics, slower):

```bash
python llm_training/player_summary_advanced.py \
  --test_only \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./player_summary_model \
  --generate_all
# Then use generated_summaries.jsonl: each line has id + generated + timing (+ peak_gpu_mb on CUDA)
```

**B. Run metrics** (PARENT, BLEU, ROUGE, chrF++, numeric coverage; optional BERTScore):

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

## Compare two presets (Mistral vs Phi-3-mini)

With **CUDA PyTorch** and both requirement files installed, from the repo root:

- **Windows:** `.\scripts\run_compare_presets.ps1` (add `-Epochs 1` for a faster trial; `-SkipTrain` if adapters already exist in `outputs/compare_presets/`).
- **Linux / macOS:** `bash scripts/run_compare_presets.sh` (`EPOCHS=1` or `SKIP_TRAIN=1` as needed).

This trains each preset into separate adapter folders, writes `mistral_val_predictions.jsonl` and `phi3_mini_val_predictions.jsonl`, and produces `eval_report_mistral.json` and `eval_report_phi3_mini.json` under `outputs/compare_presets/`.
