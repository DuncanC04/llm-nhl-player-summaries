# Player Summary Generator

Fine-tune AI models to generate player summaries from statistics. Two model options:
- **Mistral-7B (Advanced)**: High-quality summaries, requires GPU (Recommended)
- **MiniGPT (Simple)**: Lightweight model, lower accuracy, runs on CPU

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

# Generate summaries for all players
python llm_training/player_summary_advanced.py --test_only --generate_all
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

### Keras Compatibility

If you see Keras 3 errors:
```bash
python utils/fix_keras_compatibility.py
```

## Model Comparison

| Feature | Advanced Model (Mistral-7B) | Simple Model (MiniGPT) |
|---------|----------------------------|------------------------|
| Status | ✅ Recommended | ⚠️ Lower Accuracy |
| Base Model | Mistral-7B | Custom Transformer |
| Parameters | ~7B | ~5M |
| GPU Required | Yes (8GB+) | No (CPU sufficient) |
| Training Time | 30-60 min | 30-60 min (CPU) |
| Model Size | ~100MB (LoRA) | ~20MB |
| Quality | Excellent | Limited (model too small) |

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

## Simple Model: MiniGPT (Lower Accuracy)

The MiniGPT model (`llm_training/player_summary_minigpt.py`) is a lightweight alternative that runs on CPU. **It works but produces lower quality summaries** due to the small model size (~5M parameters).

### When to Use MiniGPT

- You don't have access to a GPU
- You want faster training/inference
- You can accept lower accuracy for experimental purposes
- You're testing the pipeline before investing in GPU resources

### Limitations

- **Lower accuracy**: The small model size (~5M parameters) limits its ability to generate high-quality summaries
- Output quality is significantly worse than the advanced model
- May produce less coherent or relevant summaries
- Not recommended for production use

### Usage

```bash
# Train MiniGPT model
python llm_training/player_summary_minigpt.py

# Test MiniGPT model (default: ./models/minigpt/player_summary_minigpt.keras)
python llm_training/player_summary_minigpt.py --test_only

# Custom training parameters
python llm_training/player_summary_minigpt.py \
    --epochs 30 \
    --batch_size 64 \
    --vocab_size 15000 \
    --maxlen 256 \
    --output_dir ./models/minigpt/player_summary_minigpt.keras
```

**Recommendation:** Use the **Mistral-7B (Advanced)** model for best results. Only use MiniGPT if you specifically need a CPU-only solution and can accept lower quality output.

---

## License

MIT License - see `LICENSE` file for details.
