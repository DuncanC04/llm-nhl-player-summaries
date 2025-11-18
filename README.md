# Player Summary Generator

Generate AI-powered player summaries from statistics using either a lightweight MiniGPT model or a fine-tuned Mistral-7B model.

## Models

### 1. Advanced Model (Mistral-7B)
- **File**: `llm_training/player_summary_advanced.py`
- **Architecture**: Mistral-7B with QLoRA fine-tuning
- **Requirements**: NVIDIA GPU with 12GB+ VRAM recommended
- **Training Time**: 30-60 minutes on RTX 3060
- **Quality**: High-quality, natural language summaries

### 2. Simple Model (MiniGPT)
- **File**: `llm_training/player_summary_minigpt.py`
- **Architecture**: Small transformer (~5M parameters)
- **Requirements**: CPU or basic GPU
- **Training Time**: 30-60 minutes on CPU, 5-15 minutes on GPU
- **Quality**: Good summaries, faster training

## Quick Start

### 1. Setup Environment

**Windows:**
```cmd
python setup.py
```

**Linux/Mac:**
```bash
python setup.py
```

Or manually:
```bash
python -m venv llm_env
llm_env\Scripts\activate.bat  # Windows CMD
# or: source llm_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Prepare Data

Make sure you have the training data at:
```
Data/out/aiTop10Stats_complete.jsonl
```

If you need to generate the training data from CSV files, use the data preparation scripts:
```bash
# Generate JSONL file with top 10 stats for each player
python scripts/generate_top10_stats_jsonl.py \
    --skaters Data/skaters_24_25.csv \
    --goalies Data/goalies_24_25.csv \
    --output Data/out/aiTop10Stats_complete.jsonl
```

### 3. Train a Model

**Advanced Model (Mistral-7B):**
```bash
python llm_training/player_summary_advanced.py
```

**Simple Model (MiniGPT):**
```bash
python llm_training/player_summary_minigpt.py
```

### 4. Test a Model

**Advanced Model:**
```bash
# Test on default 3 examples
python llm_training/player_summary_advanced.py --test_only --output_dir ./player_summary_model

# Test on more players (e.g., 10 examples)
python llm_training/player_summary_advanced.py --test_only --num_test 10

# Test on all validation examples
python llm_training/player_summary_advanced.py --test_only --num_test -1
```

**Simple Model:**
```bash
python llm_training/player_summary_minigpt.py --test_only --output_dir ./player_summary_minigpt.keras
```

## Advanced Usage

### Advanced Model Options

```bash
# Custom training parameters
python llm_training/player_summary_advanced.py \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --model_name mistralai/Mistral-7B-v0.1

# Test on more players (10 examples)
python llm_training/player_summary_advanced.py --test_only --num_test 10

# Generate summaries for all players
python llm_training/player_summary_advanced.py --generate_all

# Use a different model
python llm_training/player_summary_advanced.py --model_name microsoft/phi-2
```

### Simple Model Options

```bash
# Custom training parameters
python llm_training/player_summary_minigpt.py \
    --epochs 30 \
    --batch_size 64 \
    --vocab_size 15000 \
    --maxlen 256
```

## Command-Line Arguments

### Advanced Model

- `--data_path`: Path to JSONL data file (default: auto-detect)
- `--model_name`: Model to fine-tune (default: mistralai/Mistral-7B-v0.1)
- `--output_dir`: Directory to save model (default: ./player_summary_model)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--test_only`: Only test an existing model (skip training)
- `--num_test`: Number of test examples (default: 3, use -1 for all validation examples)
- `--generate_all`: Generate summaries for all players
- `--no_4bit`: Disable 4-bit quantization

### Simple Model

- `--data_path`: Path to JSONL data file (default: auto-detect)
- `--output_dir`: Output path for trained model (default: ./player_summary_minigpt.keras)
- `--test_only`: Only test an existing model (skip training)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Training batch size (default: 32)
- `--vocab_size`: Vocabulary size (default: 10000)
- `--maxlen`: Maximum sequence length (default: 128)

## Requirements

- Python 3.8+
- **Advanced Model**: 
  - NVIDIA GPU with 8GB+ VRAM (recommended: 12GB+)
  - PyTorch with CUDA support (see installation below)
- **Simple Model**: 
  - CPU or basic GPU sufficient
  - TensorFlow/Keras

See `requirements.txt` for full dependency list.

### Fixing Keras 3 / Transformers Compatibility

If you encounter errors about Keras 3 not being supported by transformers:

```bash
# Run the compatibility fix script
python fix_keras_compatibility.py

# Or manually install tf-keras
pip install tf-keras>=2.15.0
```

The advanced model script automatically sets `TF_USE_LEGACY_KERAS=1` to use the backwards-compatible tf-keras with transformers, while the simple model continues using Keras 3.

### Installing PyTorch with CUDA Support

If you have an NVIDIA GPU but PyTorch shows "CUDA available: False":

**Windows - PowerShell (Recommended):**
```powershell
# Run the PowerShell script
.\install_pytorch_cuda.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then run the script again
```

**Windows - Command Prompt:**
```cmd
install_pytorch_cuda.bat
```

**Manual Installation (if scripts don't work):**
```cmd
# 1. Activate virtual environment
llm_env\Scripts\activate.bat

# 2. Uninstall CPU-only version
python -m pip uninstall -y torch torchvision torchaudio

# 3. Install CUDA version (CUDA 12.1)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Linux/Mac:**
```bash
chmod +x install_pytorch_cuda.sh
./install_pytorch_cuda.sh
```

**Important:** 
- Make sure you're using the virtual environment's Python, not system Python
- Restart your terminal/Python environment after installing PyTorch with CUDA
- The installation downloads ~2.5GB, so it may take several minutes

## Data Format

Input data should be in JSONL format with the following structure:

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

## Troubleshooting

### PowerShell Execution Policy Error

If you get errors activating the virtual environment in PowerShell:

```powershell
# Use CMD to run batch file
cmd /c llm_env\Scripts\activate.bat

# Or change execution policy (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Permission Denied Error

If you see permission errors with the virtual environment:

```bash
# Close all Python processes
taskkill /F /IM python.exe /T  # Windows

# Delete and recreate virtual environment
rmdir /s /q llm_env  # Windows
rm -rf llm_env       # Linux/Mac
python -m venv llm_env
```

### Out of Memory (OOM) Errors

For the advanced model, reduce batch size:
```bash
python llm_training/player_summary_advanced.py --batch_size 2
```

### Tokenizer Loading Error

Update transformers library:
```bash
pip install --upgrade transformers>=4.40.0
```

## Performance Evaluation

Both models now automatically track and report timing metrics:

### Training Time

After training completes, you'll see a summary like:
```
================================================================================
TRAINING TIME METRICS
================================================================================
Total training time: 1845.32 seconds (30.76 minutes)
Time per epoch: 615.11 seconds (10.25 minutes)
================================================================================
```

### Generation Time

When testing models, timing information is displayed:

1. **Per-summary timing**: Each generated summary shows its generation time
2. **Summary statistics**: After testing, you'll see:
   ```
   ================================================================================
   GENERATION TIME STATISTICS
   ================================================================================
   Number of summaries generated: 3
   Average time per summary: 2.345 seconds
   Min time: 2.101 seconds
   Max time: 2.567 seconds
   Total time: 7.035 seconds
   ================================================================================
   ```

3. **Batch generation**: When using `--generate_all`, you'll see:
   ```
   ================================================================================
   BATCH GENERATION TIME STATISTICS
   ================================================================================
   Total players processed: 500
   Total time: 1172.50 seconds (19.54 minutes)
   Average time per summary: 2.345 seconds
   Min time: 2.101 seconds
   Max time: 2.567 seconds
   Summaries per minute: 25.6
   ================================================================================
   ```

### Example Usage

```bash
# Train and see training time
python llm_training/player_summary_advanced.py --num_epochs 3

# Test and see generation times
python llm_training/player_summary_advanced.py --test_only --num_test 10

# Generate all summaries with timing stats
python llm_training/player_summary_advanced.py --test_only --generate_all
```

## Model Comparison

| Feature | Advanced Model | Simple Model |
|---------|---------------|--------------|
| Base Model | Mistral-7B | Custom Transformer |
| Parameters | ~7B | ~5M |
| GPU Required | Yes (12GB+) | No |
| Training Time | 30-60 min | 30-60 min (CPU) |
| Model Size | ~100MB (LoRA) | ~20MB |
| Quality | Excellent | Good |
| Speed | Fast inference | Very fast inference |

## License

This code is provided as-is. Please respect:
- Model licenses (Mistral-7B: Apache 2.0)
- Your data source licenses
- Fair use guidelines for generated content

