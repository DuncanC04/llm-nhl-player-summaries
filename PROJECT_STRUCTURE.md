# Project Structure

```
CS_Fall_Research/
├── README.md                 # Main documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script for virtual environment
│
├── llm_training/             # Training scripts
│   ├── player_summary_advanced.py  # Mistral-7B model (Recommended)
│   └── player_summary_minigpt.py   # MiniGPT model (Lower accuracy)
│
├── scripts/                  # Data processing scripts
│   ├── generate_top10_stats_jsonl.py      # Generate training data from CSV
│   └── generate_player_summaries.py       # Generate player summaries
│
├── utils/                    # Utility scripts
│   ├── fix_keras_compatibility.py  # Fix Keras 3 compatibility
│   └── test_compatibility.py       # Test compatibility
│
├── Data/                     # Data files (not in git)
│   ├── *.csv                # Input CSV files
│   └── out/                 # Generated JSONL files
│
├── models/                   # Trained model outputs (not in git)
│   ├── minigpt/             # MiniGPT model files
│   │   ├── player_summary_minigpt.keras
│   │   └── player_summary_minigpt_vocab.json
│   └── ...                  # Other model outputs
│
├── results/                  # Training checkpoints (not in git)
│   └── checkpoint-*/        # Model checkpoints during training
│
├── player_summary_model/     # Trained advanced model (not in git)
│   └── adapter_*.safetensors # LoRA adapter weights
│
└── llm_env/                  # Virtual environment (not in git)
    └── ...
```

## Directory Descriptions

### `/llm_training`
Contains the main training scripts for both models:
- **player_summary_advanced.py**: Mistral-7B with QLoRA (recommended)
- **player_summary_minigpt.py**: MiniGPT lightweight model (lower accuracy)

### `/scripts`
Data processing and preparation scripts:
- **generate_top10_stats_jsonl.py**: Converts CSV files to training JSONL format
- **generate_player_summaries.py**: Utility for generating summaries

### `/utils`
Utility and helper scripts:
- **fix_keras_compatibility.py**: Fixes Keras 3/Transformers compatibility issues
- **test_compatibility.py**: Tests library compatibility

### Excluded from Git (via .gitignore)
- `llm_env/` - Virtual environment
- `Data/` - Data files (CSV, JSONL)
- `models/` - All trained model outputs
- `results/` - Training checkpoints
- `player_summary_model/` - Trained advanced models
- `*.keras`, `*.safetensors`, `*.vocab.json` - Model files
- `__pycache__/` - Python cache

