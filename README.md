# Player Summary Generator

Can large language models generate accurate, fluent player summaries directly from structured hockey statistics?

Fine-tune a **base language model** with QLoRA to generate player summaries from statistics (NVIDIA GPU required). The training code is modular under **`llm_training/player_summary/`**; **[`player_summary_advanced.py`](llm_training/player_summary_advanced.py)** is the CLI entry point.

**Cloned from GitHub?** Use **[docs/getting-started.md](docs/getting-started.md)** (environment, JSONL data, training, evaluation). For the **table-to-text** pattern and how to apply this to your own domain, see **[docs/table-to-text.md](docs/table-to-text.md)**. Schema: **[docs/custom-dataset.md](docs/custom-dataset.md)**. Presets, metrics, and extending models: **[docs/models-and-evaluation.md](docs/models-and-evaluation.md)**.

Continuous integration runs a **no-GPU smoke test** of the evaluation stack (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Model presets (`--model_preset`)

Four presets ship out of the box. All use QLoRA (LoRA rank 16, alpha 32) and the same JSONL training/evaluation pipeline. Results below are from 3-epoch fine-tuning on the hockey dataset (45 validation examples):

| Preset | Base model | PARENT↑ | BLEU↑ | ROUGE-1↑ | Peak GPU |
|--------|-----------|---------|-------|----------|----------|
| `gpt2` | GPT-2 (124M) | 0.035 | 0.70 | 0.218 | ~260 MB |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | 0.064 | 2.30 | 0.297 | ~2.2 GB |
| `phi-3-mini` | Phi-3-mini-4k-instruct | 0.028 | 0.55 | 0.236 | ~3.0 GB |
| **`mistral`** *(default)* | Mistral-7B-v0.1 | **0.083** | **5.19** | **0.333** | ~4.9 GB |

Full per-example results and adapter configs: [`outputs/compare_presets/`](outputs/compare_presets/).

Override the base checkpoint with **`--model_name`** (any compatible causal LM id). LoRA target layers are resolved automatically per preset. See [docs/models-and-evaluation.md](docs/models-and-evaluation.md) to register a new preset.

**Shuffle before train/val split:** JSONL rows are shuffled with **`--shuffle_seed 42`** by default. Use **`--no_shuffle`** to keep file order, or **`--shuffle_seed N`** for a different split.

## Quick Start

### 1. Setup

```bash
python -m venv llm_env
llm_env\Scripts\activate.bat   # Windows
# source llm_env/bin/activate  # Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-eval.txt
```

If `python -c "import torch; print(torch.cuda.is_available())"` prints `False`, install a CUDA-enabled PyTorch wheel (see [PyTorch CUDA Support](#pytorch-cuda-support) below).

### 2. Prepare data

Data files are not included. Provide your own JSONL, or generate from CSVs:

```bash
python scripts/generate_top10_stats_jsonl.py \
    --skaters Data/skaters_24_25.csv \
    --goalies Data/goalies_24_25.csv \
    --output Data/out/aiTop10Stats_complete.jsonl
```

JSONL format: see [Data Format](#data-format) below and [docs/custom-dataset.md](docs/custom-dataset.md). To use this pipeline on your own domain (product descriptions, lab reports, etc.), see [docs/table-to-text.md](docs/table-to-text.md).

### 3. Train

```bash
# Mistral-7B (default, best quality, ~5 GB VRAM)
python llm_training/player_summary_advanced.py --data_path Data/out/aiTop10Stats_complete.jsonl

# Qwen3-1.7B (good quality, ~2 GB VRAM)
python llm_training/player_summary_advanced.py --model_preset qwen3-1.7b --output_dir ./adapter_qwen3

# Phi-3-mini (chat template, ~3 GB VRAM)
python llm_training/player_summary_advanced.py --model_preset phi-3-mini --output_dir ./adapter_phi3

# GPT-2 (smallest, CPU-feasible, lowest quality)
python llm_training/player_summary_advanced.py --model_preset gpt2 --output_dir ./adapter_gpt2
```

### 4. Evaluate

```bash
# Export validation predictions (match --model_preset and --output_dir from training)
python llm_training/player_summary_advanced.py --test_only --num_test -1 \
  --export_predictions outputs/mistral_val_predictions.jsonl

# Run metrics (PARENT, BLEU, ROUGE, chrF++, numeric_coverage)
python -m evaluation.run_eval \
  --gold Data/out/aiTop10Stats_complete.jsonl \
  --pred outputs/mistral_val_predictions.jsonl \
  --out outputs/eval_report.json
```

**Compare all presets** (train, export, and evaluate all 4 in one shot):

- **Windows:** `.\scripts\run_compare_presets.ps1` (add `-Epochs 1` for a quick trial; `-SkipTrain` to re-evaluate existing adapters)
- **Linux/macOS:** `bash scripts/run_compare_presets.sh` (`EPOCHS=1` or `SKIP_TRAIN=1` as needed)

Artifacts land in `outputs/compare_presets/` — `eval_report_<preset>.json` and `<preset>_val_predictions.jsonl` for each model.

## Configuration

**Common arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_preset` | `mistral` | `gpt2`, `qwen3-1.7b`, `phi-3-mini`, or `mistral` |
| `--model_name` | preset default | Override with any Hugging Face causal LM id |
| `--data_path` | `Data/out/aiTop10Stats_complete.jsonl` | Path to your JSONL file |
| `--output_dir` | `./player_summary_model` | Where the LoRA adapter is saved |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Training batch size |
| `--learning_rate` | `2e-4` | Learning rate |
| `--shuffle_seed` | `42` | RNG seed for train/val split |
| `--no_shuffle` | — | Keep JSONL row order (no shuffle) |
| `--test_only` | — | Skip training; load existing adapter |
| `--num_test` | `3` | Test examples (`-1` for all) |
| `--export_predictions` | — | Write validation JSONL for `run_eval` |
| `--generate_all` | — | Generate for the full dataset |

**Code layout:** `llm_training/player_summary/` — `cli.py`, `data_pipeline.py`, `prompts.py`, `model_setup.py`, `training_loop.py`, `inference.py`, `evaluation_run.py`, `presets/`. To add a new base model, add a preset class under `presets/` and register it in `presets/registry.py` (details: [docs/models-and-evaluation.md](docs/models-and-evaluation.md)).

## Data Format

One UTF-8 JSON object per line:

```json
{
  "name": "Connor McDavid",
  "team": "EDM",
  "position": "C",
  "topStats": [
    {"stat": "points", "value": 132.0, "pctl": 99},
    {"stat": "assists", "value": 89.0, "pctl": 99}
  ],
  "summary": "Connor McDavid dominates offensively..."
}
```

Full schema and stable IDs: [docs/custom-dataset.md](docs/custom-dataset.md).

---

## Evaluation (automatic + human)

### Run automatic metrics

```bash
python -m evaluation.run_eval \
  --gold Data/out/aiTop10Stats_complete.jsonl \
  --pred outputs/mistral_val_predictions.jsonl \
  --out outputs/eval_report.json
```

Default metrics: `parent`, `bleu`, `chrf`, `rouge`, `numeric_coverage`. Add `bertscore` to `--metrics` for semantic similarity (downloads weights on first run).

PARENT is implemented in-repo (`evaluation/parent_metric.py`, word-overlap variant, Dhingra et al. 2019). It compares generated text against the structured table, penalizing hallucinated values more than BLEU/ROUGE.

### Human rubric (optional)

1. Copy `evaluation/human_rubric_template.csv` and fill ratings following `evaluation/RATER_INSTRUCTIONS.txt`.
2. Merge with automatic scores:

```bash
python -m evaluation.merge_human \
  --human my_ratings.csv \
  --report outputs/eval_report.json \
  --out outputs/eval_plus_human.json
```

---

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (`gpt2` preset can run on CPU for small datasets)
- 8 GB+ VRAM for `mistral`; 2–5 GB for smaller presets (see table above)

### PyTorch CUDA Support

If `torch.cuda.is_available()` is `False`:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

### Keras / Transformers compatibility

```bash
python utils/fix_keras_compatibility.py
```

---

## License

MIT License — see `LICENSE` for details.
