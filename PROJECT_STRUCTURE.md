# Project Structure

```
CS_Fall_Research/
├── README.md                 # Main documentation (quick start + evaluation summary)
├── docs/                     # Detailed guides (GitHub-friendly)
│   ├── README.md             # Index of doc pages
│   ├── getting-started.md    # Clone → train Mistral → evaluate
│   ├── custom-dataset.md     # JSONL schema, custom data, stable IDs
│   └── models-and-evaluation.md  # Metrics, adding future models
├── LICENSE                   # MIT License
├── requirements.txt          # Mistral / PyTorch stack
├── requirements-eval.txt     # Optional: PARENT-related automatic metrics (BLEU, ROUGE, …)
├── setup.py                  # Interactive venv + dependency installer
│
├── evaluation/               # Table-to-text metrics (model-agnostic)
│   ├── run_eval.py           # CLI: gold + predictions → eval_report.json
│   ├── merge_human.py        # Join human rubric CSV with eval_report
│   ├── parent_metric.py      # PARENT (Dhingra et al., 2019)
│   ├── automatic_metrics.py  # BLEU, chrF++, ROUGE, BERTScore, numeric coverage
│   ├── jsonl_table.py        # Records + stable_example_id for alignment
│   ├── human_rubric_template.csv
│   └── RATER_INSTRUCTIONS.txt
│
├── .github/workflows/
│   └── ci.yml                # Smoke test evaluation (no GPU)
│
├── llm_training/
│   └── player_summary_advanced.py  # Mistral-7B + QLoRA
│
├── scripts/
│   ├── generate_top10_stats_jsonl.py   # CSV → JSONL (hockey example)
│   ├── generate_player_summaries.py
│   └── smoke_eval.py                 # CI: toy gold/pred → run_eval
│
├── utils/
│   ├── fix_keras_compatibility.py
│   └── test_compatibility.py
│
├── Data/                     # Data files (not in git; you provide CSV/JSONL)
├── models/                   # Trained outputs (not in git; legacy name)
├── results/                  # Training checkpoints (not in git)
├── player_summary_model/     # Default Mistral LoRA export (not in git)
├── outputs/                  # Local prediction/eval exports (not in git)
│
└── llm_env/                  # Virtual environment (not in git)
```

## Directory descriptions

### `/evaluation`

Shared **scoring pipeline** for any model that emits the standard predictions JSONL (`id`, `generated`, optional timing / VRAM). See `docs/models-and-evaluation.md`.

### `/docs`

On GitHub, open these Markdown files in the browser for full setup, custom datasets, and how to plug in additional models later.

### `/llm_training`

**player_summary_advanced.py**: Mistral-7B with QLoRA; supports `--export_predictions` and `--generate_all` for eval.

### Excluded from Git (via .gitignore)

- `llm_env/`, `Data/`, `models/`, `results/`, `player_summary_model/`, `outputs/`
- Large model weights and `*.jsonl` / `*.csv` patterns as configured
- See `.gitignore` for the full list
