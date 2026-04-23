# Models and evaluation

## Training entry point and code layout

| Preset | Default base model | CLI |
|--------|-------------------|-----|
| **`mistral`** (default) | `mistralai/Mistral-7B-v0.1` | `python llm_training/player_summary_advanced.py` |
| **`phi-3-mini`** | `microsoft/Phi-3-mini-4k-instruct` | same script + `--model_preset phi-3-mini` |

- **Entry script:** `llm_training/player_summary_advanced.py` (delegates to `llm_training/player_summary/cli.py`).
- **Package:** `llm_training/player_summary/` — data loading and shuffle, prompts, model/LoRA setup, `SFTTrainer`, inference, and validation / full export.
- **Presets:** `llm_training/player_summary/presets/` — one module per family (e.g. `mistral.py`, `phi3_mini.py`) plus `registry.py` and shared `lora.py` (`resolve_lora_targets`).

**Hardware:** NVIDIA GPU with CUDA; Mistral-7B in 4-bit typically wants **8GB+ VRAM** (12GB+ comfortable). Phi-3-mini is smaller; still use the same eval export flags.

**Shuffle:** JSONL rows are shuffled before the train/validation split by default (`--shuffle_seed 42`). Use `--no_shuffle` or change `--shuffle_seed` as needed. **`stable_example_id`** in gold data is content-based, so metric alignment does not depend on row order.

Training data: JSONL schema in [Custom dataset](custom-dataset.md).

## Evaluation stack (table-to-text)

Aligned with Dhingra et al. (2019) — [PARENT](https://arxiv.org/abs/1906.01081) and reference-based metrics:

| Component | Location |
|-----------|----------|
| PARENT (word-overlap variant) | `evaluation/parent_metric.py` |
| BLEU, chrF++, ROUGE, optional BERTScore | `evaluation/automatic_metrics.py` |
| Numeric sanity check | `evaluation/automatic_metrics.py` → `numeric_coverage` |
| Driver CLI | `python -m evaluation.run_eval` |
| Human Likert rubric | `evaluation/human_rubric_template.csv`, `RATER_INSTRUCTIONS.txt` |
| Merge human + automatic | `python -m evaluation.merge_human` |

**Interpretation:** When reference summaries diverge from the table, PARENT is designed to track human judgments better than BLEU/ROUGE alone; still report n-gram metrics for comparability.

## Exporting predictions (any future model)

To compare a **new** model on the same gold file:

1. For each gold line, compute `id = stable_example_id(example)` (see `evaluation/jsonl_table.py`).
2. Generate text from your model given the same inputs you use at training time.
3. Write JSONL lines minimally as:

```json
{"id": "<stable id>", "generated": "<model output>"}
```

Optional for efficiency reporting:

```json
{
  "id": "...",
  "generated": "...",
  "generation_time_s": 0.42,
  "num_output_tokens": 120,
  "peak_gpu_mb": 8192.0
}
```

4. Run `evaluation.run_eval --gold your.jsonl --pred your_preds.jsonl --out report.json`.

## Adding another base model (new preset)

1. Add a preset class under `llm_training/player_summary/presets/` (see `mistral.py` / `phi3_mini.py` for examples). It should define:
   - `id` (CLI string), `default_model_id`, `lora_target_candidates`
   - `build_training_text(example, prompt, tokenizer)` — full SFT line (plain continuation or chat-formatted string)
   - `build_generation_inputs(tokenizer, prompt)` — tensor dict + prompt length for `model.generate`
   - `extra_generation_stop_markers()` — optional strings to trim from decoded output (e.g. chat turn markers)
2. Register the class in `presets/registry.py` (`_PRESETS`). `PRESET_IDS` and argparse `--model_preset` update automatically.
3. Reuse **evaluation** unchanged: same `--gold` JSONL, same prediction JSONL contract from `--export_predictions` / `--generate_all`.

If you change how **table fields** appear in the prompt, update **`evaluation/jsonl_table.py`** (`records_from_example`, and `stable_example_id` if identity rules change) so PARENT and IDs stay aligned with what the model sees.

Keeping **one JSONL schema** and **one prediction format** lets you run the same PARENT / BLEU / human pipeline for every model generation.
