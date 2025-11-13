# Quick Start Guide - Advanced Model

## Prerequisites
✅ CUDA is installed and working (verify with `python verify_cuda.py`)

## Running the Advanced Model

### 1. Activate Virtual Environment
```powershell
# Windows PowerShell or CMD
llm_env\Scripts\activate.bat
```

### 2. Train the Model
```bash
# Basic training (uses defaults)
python llm_training/player_summary_advanced.py

# Custom training parameters
python llm_training/player_summary_advanced.py \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### 3. Test an Existing Model
```bash
python llm_training/player_summary_advanced.py \
    --test_only \
    --output_dir ./player_summary_model
```

## Training Parameters

**Recommended for RTX 3060 Ti (8GB VRAM):**
- `--batch_size 2` (or even 1 if you get OOM errors)
- `--num_epochs 3-5`
- `--learning_rate 2e-4` (default)
- Keep `--max_seq_length 512` (default)

**If you get Out of Memory (OOM) errors:**
1. Reduce batch size: `--batch_size 1`
2. Reduce sequence length: `--max_seq_length 256`
3. Increase gradient accumulation (in code)

## Expected Training Time
- RTX 3060 Ti: **30-60 minutes** for 3 epochs
- Training progress will be displayed with loss values
- Model checkpoints saved every 50 steps

## Output
- Trained model saved to: `./player_summary_model/`
- Training results: `./results/`
- Model files: ~100MB (LoRA adapters only)

## Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size to 1
- Reduce max_seq_length to 256
- Close other GPU applications

**Slow Training:**
- Normal for first epoch (model loading)
- Subsequent epochs should be faster
- Check GPU utilization with `nvidia-smi`

**Model Not Learning:**
- Check training loss (should decrease)
- Try more epochs (5-7)
- Verify data quality

