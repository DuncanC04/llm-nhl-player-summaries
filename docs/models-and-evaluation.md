# Models and evaluation

## Current training entry point

| Model | Script | Hardware | Notes |
|-------|--------|----------|--------|
| **Mistral-7B + QLoRA** | `llm_training/player_summary_advanced.py` | NVIDIA GPU, 8GB+ VRAM | `--export_predictions` / `--generate_all` for eval JSONL + `peak_gpu_mb` on CUDA |

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

## Adding a small language model later

1. Add a script under `llm_training/` (or a subpackage) that loads your weights, reads JSONL, and formats prompts consistently with Mistral training (or documents a new prompt format and updates `jsonl_table` if the table changes).
2. Reuse **evaluation** unchanged: same `--gold` JSONL, same prediction JSONL contract.
3. Register the script in the main README and optionally add a row to the table at the top of this file.

Keeping **one JSONL schema** and **one prediction format** lets you run the same PARENT / BLEU / human pipeline for every model generation.
