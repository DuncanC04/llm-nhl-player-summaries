# Player Summary Generator

Fine-tune **Mistral-7B** with QLoRA to generate player summaries from statistics (GPU required).

**Cloned from GitHub?** Use the step-by-step guide in **[docs/getting-started.md](docs/getting-started.md)** (environment, your own JSONL data, training, and full evaluation). More detail: **[docs/custom-dataset.md](docs/custom-dataset.md)** (schema), **[docs/models-and-evaluation.md](docs/models-and-evaluation.md)** (metrics + adding future models).

Continuous integration runs a **no-GPU smoke test** of the evaluation stack (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Recommended: Mistral-7B (Advanced)

The advanced model uses Mistral-7B with QLoRA fine-tuning to generate high-quality player summaries. **This is the recommended option** for best results.

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
```

#### 4. Test Model

```bash
# Test on 3 examples (default)
python llm_training/player_summary_advanced.py --test_only

# Test on more examples
python llm_training/player_summary_advanced.py --test_only --num_test 10

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
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--test_only`: Skip training, only test existing model
- `--num_test`: Number of test examples (default: 3, use -1 for all)
- `--generate_all`: Generate summaries for all players
- `--export_predictions`: Path to JSONL written after validation testing (for `evaluation.run_eval`)

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

## Mistral-7B at a glance

| Feature | Value |
|---------|--------|
| Base model | Mistral-7B |
| Fine-tuning | QLoRA (LoRA adapters) |
| GPU | NVIDIA, 8GB+ VRAM (12GB+ recommended) |
| Typical training time | ~30–60 minutes |
| Adapter size on disk | ~100MB (order of magnitude) |

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
