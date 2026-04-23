# Models and evaluation

## Training entry point and code layout

| Preset | Base model | Notes |
|--------|-----------|-------|
| `gpt2` | `openai-community/gpt2` (124M) | Smallest; plain continuation; CPU-feasible |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | Good quality/VRAM trade-off |
| `phi-3-mini` | `microsoft/Phi-3-mini-4k-instruct` (3.8B) | Chat-template for train and inference |
| **`mistral`** *(default)* | `mistralai/Mistral-7B-v0.1` | Best scores; needs ~8 GB+ VRAM in 4-bit |

- **Entry script:** `llm_training/player_summary_advanced.py` (delegates to `llm_training/player_summary/cli.py`).
- **Package:** `llm_training/player_summary/` — data loading and shuffle, prompts, model/LoRA setup, `SFTTrainer`, inference, and validation/full export.
- **Presets:** `llm_training/player_summary/presets/` — one module per family (`gpt2.py`, `mistral.py`, `phi3_mini.py`, `qwen35_27b_dflash.py`) plus `registry.py` and shared `lora.py`.

**Hardware:** NVIDIA GPU with CUDA. GPT-2 can run on CPU for small datasets. For Mistral-7B in 4-bit, 8 GB+ VRAM (12 GB+ comfortable).

**Shuffle:** JSONL rows are shuffled before the train/validation split by default (`--shuffle_seed 42`). Use `--no_shuffle` or change `--shuffle_seed` as needed. `stable_example_id` in gold data is content-based, so metric alignment does not depend on row order.

Training data schema: [custom-dataset.md](custom-dataset.md).

## Benchmark results (hockey dataset, 45 val examples, 3 epochs, LoRA r=16)

| Preset | PARENT↑ | BLEU↑ | chrF++↑ | ROUGE-1↑ | Tokens/s | Peak GPU |
|--------|---------|-------|---------|----------|----------|----------|
| `gpt2` | 0.035 | 0.70 | 24.2 | 0.218 | 26.2 | ~260 MB |
| `qwen3-1.7b` | 0.064 | 2.30 | 28.3 | 0.297 | 7.1 | ~2.2 GB |
| `phi-3-mini` | 0.028 | 0.55 | 26.0 | 0.236 | 9.2 | ~3.0 GB |
| `mistral` | **0.083** | **5.19** | **30.7** | **0.333** | 6.8 | ~4.9 GB |

Per-example reports and prediction files: `outputs/compare_presets/`. Reproduce with `scripts/run_compare_presets.ps1` (Windows) or `scripts/run_compare_presets.sh` (Linux/macOS).

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

**Interpretation:** When reference summaries diverge from the table, PARENT tracks human judgments better than BLEU/ROUGE alone; still report n-gram metrics for comparability across systems.

## Exporting predictions (any model)

To compare a **new** model on the same gold file:

1. For each gold line, compute `id = stable_example_id(example)` (see `evaluation/jsonl_table.py`).
2. Generate text from your model given the same prompt used at training time.
3. Write JSONL lines minimally as:

```json
{"id": "<stable id>", "generated": "<model output>"}
```

Optional efficiency fields:

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

The built-in presets write this format automatically via `--export_predictions` or `--generate_all`.

## Adding a new base model (new preset)

1. Add a preset class under `llm_training/player_summary/presets/` (see `mistral.py` or `phi3_mini.py` for examples). Define:
   - `id` (CLI string), `default_model_id`, `lora_target_candidates`
   - `build_training_text(example, prompt, tokenizer)` — full SFT string (plain continuation or chat-formatted)
   - `build_generation_inputs(tokenizer, prompt)` — tensor dict + prompt length for `model.generate`
   - `extra_generation_stop_markers()` — strings to trim from decoded output (e.g. chat turn markers)
2. Register the class in `presets/registry.py` (`_PRESETS`). `PRESET_IDS` and argparse `--model_preset` update automatically.
3. Run evaluation unchanged: same `--gold` JSONL, same prediction JSONL format.

If you change how **table fields** appear in the prompt, update `evaluation/jsonl_table.py` (`records_from_example`, and `stable_example_id` if identity rules change) so PARENT and IDs stay aligned.

Keeping **one JSONL schema** and **one prediction format** lets you run the same PARENT/BLEU/human pipeline for every model — which is exactly the point if you are comparing architectures or applying this to a new domain.
