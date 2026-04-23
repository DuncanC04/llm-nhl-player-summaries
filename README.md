# Player Summary Generator

Fine-tune a **base language model** with QLoRA to generate player summaries from statistics (NVIDIA GPU required). The training code is modular under **`llm_training/player_summary/`**; **[`player_summary_advanced.py`](llm_training/player_summary_advanced.py)** is the CLI entry point.

**Cloned from GitHub?** Use **[docs/getting-started.md](docs/getting-started.md)** (environment, JSONL data, training, evaluation). For the **table-to-text** pattern, see **[docs/table-to-text.md](docs/table-to-text.md)**. Schema: **[docs/custom-dataset.md](docs/custom-dataset.md)**. Presets, metrics, and extending models: **[docs/models-and-evaluation.md](docs/models-and-evaluation.md)**.

Continuous integration runs a **no-GPU smoke test** of the evaluation stack (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Model presets (`--model_preset`)

| Preset | Default Hugging Face model | Notes |
|--------|----------------------------|--------|
| `mistral` (default) | `mistralai/Mistral-7B-v0.1` | Recommended for best quality; ~8GB+ VRAM with 4-bit |
| `phi-3-mini` | `microsoft/Phi-3-mini-4k-instruct` | Smaller SLM; chat template for train and inference |

Override the checkpoint with **`--model_name`** (any compatible causal LM id). LoRA target layers are chosen automatically from each preset’s candidate list.

**Shuffle before train/val split:** By default, JSONL rows are shuffled with **`--shuffle_seed 42`** so the validation slice is not order-biased. Use **`--no_shuffle`** to keep file order. Use **`--shuffle_seed N`** for a different split.

## Recommended: Mistral-7B

**Mistral-7B** with QLoRA remains the default preset for the strongest summaries. Use **`--model_preset phi-3-mini`** to experiment with a smaller model on the same JSONL and evaluation pipeline.

### Quick Start

#### 1. Setup

```bash
# Create virtual environment and install dependencies
python setup.py

# Activate environment
llm_env\Scripts\activate.bat  # Windows
# or: source llm_env/bin/activate  # Linux/Mac
```

#### 2. Prepare Data

**Note:** Data files are not included in this repository. You'll need to provide your own CSV files or training data.

Create the training data from CSV files:
```bash
# Place your CSV files in Data/ directory first
python scripts/generate_top10_stats_jsonl.py \
    --skaters Data/skaters_24_25.csv \
    --goalies Data/goalies_24_25.csv \
    --output Data/out/aiTop10Stats_complete.jsonl
```

Or use an existing JSONL file with the structure shown in the Data Format section below.

#### 3. Train Model

```bash
python llm_training/player_summary_advanced.py

# Phi-3-mini (same flags; use the same --model_preset for --test_only later)
python llm_training/player_summary_advanced.py --model_preset phi-3-mini --output_dir ./player_summary_phi3
```

#### 4. Test Model

```bash
# Test on 3 examples (default); match the preset used at training time
python llm_training/player_summary_advanced.py --test_only

# Test on more examples
python llm_training/player_summary_advanced.py --test_only --num_test 10

# After training Phi-3 into ./player_summary_phi3
python llm_training/player_summary_advanced.py --test_only --model_preset phi-3-mini --output_dir ./player_summary_phi3

# Export validation predictions for evaluation (PARENT, BLEU, ROUGE, …)
python llm_training/player_summary_advanced.py --test_only --num_test -1 \
  --export_predictions outputs/mistral_val_predictions.jsonl

# Generate summaries for all players (also writes eval-ready id + generated + timing + peak_gpu_mb)
python llm_training/player_summary_advanced.py --test_only --generate_all
```

Then run metrics (after `pip install -r requirements-eval.txt`):

```bash
python -m evaluation.run_eval --gold Data/out/aiTop10Stats_complete.jsonl \
  --pred outputs/mistral_val_predictions.jsonl --out outputs/eval_report.json
```

**Compare Mistral vs Phi-3-mini** (train both, export validation preds, run `run_eval` twice) after CUDA PyTorch is installed:

- **Windows:** `.\scripts\run_compare_presets.ps1` (optional: `-Epochs 1` for a quick run, `-SkipTrain` to re-evaluate existing adapters under `outputs/compare_presets/`).
- **Linux/macOS:** `bash scripts/run_compare_presets.sh` (set `EPOCHS=1` or `SKIP_TRAIN=1` as needed).

Artifacts: `outputs/compare_presets/eval_report_mistral.json` and `eval_report_phi3_mini.json`.

## Configuration

### Advanced Model Options

```bash
python llm_training/player_summary_advanced.py \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --output_dir ./player_summary_model
```

**Common Arguments:**
- `--model_preset`: `mistral` or `phi-3-mini` (default: `mistral`)
- `--model_name`: Hugging Face model id (default: chosen from `--model_preset`)
- `--shuffle_seed`: RNG seed for shuffling JSONL before train/val split (default: `42`; ignored if `--no_shuffle`)
- `--no_shuffle`: Do not shuffle; first rows go to train, last rows to validation in file order
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--test_only`: Skip training, only test existing model
- `--num_test`: Number of test examples (default: 3, use -1 for all)
- `--generate_all`: Generate summaries for all players
- `--export_predictions`: Path to JSONL written after validation testing (for `evaluation.run_eval`)

**Code layout:** Implementation lives in **`llm_training/player_summary/`** (`cli.py`, `data_pipeline.py`, `prompts.py`, `model_setup.py`, `training_loop.py`, `inference.py`, `evaluation_run.py`, `presets/`). To add another base model, add a preset class and register it in `presets/registry.py` (see [docs/models-and-evaluation.md](docs/models-and-evaluation.md)).

## Requirements

- Python 3.8+
- **Advanced Model**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended)
- PyTorch with CUDA support

See `requirements.txt` for full dependencies.

## Installation Issues

### PyTorch CUDA Support

If PyTorch shows "CUDA available: False", install CUDA-enabled PyTorch:

**Windows/Linux/Mac:**
```bash
# Activate virtual environment first
llm_env\Scripts\activate.bat  # Windows
# or: source llm_env/bin/activate  # Linux/Mac

# Uninstall CPU-only version
pip uninstall -y torch torchvision torchaudio

# Install CUDA version (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Keras / Transformers compatibility

If the Transformers stack reports Keras-related import errors, try:

```bash
python utils/fix_keras_compatibility.py
```

## Hardware at a glance

| | Mistral (default) | Phi-3-mini |
|--|-------------------|------------|
| Fine-tuning | QLoRA | QLoRA |
| GPU (typical) | NVIDIA, 8GB+ VRAM (12GB+ recommended) | Often less VRAM than 7B; still use CUDA |
| Adapter size | ~100MB (order of magnitude) | Smaller base ⇒ often smaller adapter |

## Troubleshooting

**Out of Memory:** Reduce `--batch_size` to 1 or 2

**Slow Training:** Normal for first epoch; subsequent epochs should be faster

**CUDA Errors:** Verify with `python -c "import torch; print(torch.cuda.is_available())"`. If False, see PyTorch CUDA Support section above.

## Data Format

Training data should be JSONL format:
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

---

## Evaluation (automatic + human)

### 1. Install eval dependencies

```bash
# Windows
llm_env\Scripts\activate.bat
# Linux / macOS
# source llm_env/bin/activate

pip install -r requirements-eval.txt
```

PARENT is implemented in-repo (`evaluation/parent_metric.py`, word-overlap variant from Dhingra et al., 2019). See the module docstring in that file for notes on external implementations (Google Research `table_text_eval`, KaijuML/parent). Other automatic metrics use the packages above.

### 2. Export predictions JSONL

Each line must include `id` (use the same ID as gold: run Python `from evaluation.jsonl_table import stable_example_id; stable_example_id(row)` on each gold row), plus `generated` text. Optional efficiency fields: `generation_time_s`, `num_output_tokens`, and (PyTorch + CUDA) `peak_gpu_mb` per example.

`llm_training/player_summary_advanced.py` writes eval-ready JSONL via **`--export_predictions`** (validation split) or **`--generate_all`** (full dataset): `id`, `generated`, timing, `num_output_tokens`, and `peak_gpu_mb` when CUDA is available.

### 3. Run automatic metrics

From the repository root:

```bash
python -m evaluation.run_eval --gold Data/out/aiTop10Stats_complete.jsonl --pred path/to/predictions.jsonl --out eval_report.json
```

Default metrics: `parent`, `bleu`, `chrf`, `rouge`, `numeric_coverage`. Add BERTScore (downloads model weights on first run):

```bash
python -m evaluation.run_eval --gold ... --pred ... --out ... --metrics parent,bleu,chrf,rouge,bertscore,numeric_coverage
```

### 4. Human rubric

1. Copy `evaluation/human_rubric_template.csv` and follow `evaluation/RATER_INSTRUCTIONS.txt`.
2. Fill `example_id` from `eval_report.json` → `per_example` → `id`.
3. Merge with automatic scores:

```bash
python -m evaluation.merge_human --human my_ratings.csv --report eval_report.json --out eval_plus_human.json
```

The merged JSON includes `automatic_by_id`, `human_per_example`, and `combined_by_id` (aligned automatic + human fields per `example_id`).

---

## License

MIT License - see `LICENSE` file for details.
