# Getting started (GitHub clone)

This walkthrough assumes an **NVIDIA GPU** with **8GB+ VRAM** for the recommended **Mistral-7B** path.

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

You need one **JSONL** file: one JSON object per line. See [Custom dataset](custom-dataset.md) for the exact schema.

Example using the included CSV helper (place your CSVs under `Data/`):

```bash
python scripts/generate_top10_stats_jsonl.py \
  --skaters Data/skaters_24_25.csv \
  --goalies Data/goalies_24_25.csv \
  --output Data/out/my_dataset.jsonl
```

Or bring your own `my_dataset.jsonl` and pass `--data_path`.

## 3. Train Mistral-7B (QLoRA)

```bash
python llm_training/player_summary_advanced.py \
  --data_path Data/out/my_dataset.jsonl \
  --output_dir ./player_summary_model \
  --num_epochs 3
```

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
