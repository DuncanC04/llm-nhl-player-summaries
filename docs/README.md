# Documentation index

Browse these files on GitHub (or locally) for setup beyond the main [README](../README.md).

| Doc | Purpose |
|-----|---------|
| [Getting started](getting-started.md) | Clone, environment, train (Mistral or Phi-3-mini preset), shuffle/split, full evaluation |
| [Custom dataset](custom-dataset.md) | JSONL schema, building data from your own tables, stable IDs |
| [Models & evaluation](models-and-evaluation.md) | Presets, `player_summary/` package layout, metrics (PARENT, BLEU, …), registering new models |
| [Table to text](table-to-text.md) | Domain-agnostic table-to-text recipe mapped to this repo’s files |

The evaluation package lives in [`evaluation/`](../evaluation/): `run_eval`, `merge_human`, human rubric templates.
