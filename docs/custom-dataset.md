# Custom dataset (JSONL)

Training and evaluation both use the **same** JSONL format: one UTF-8 JSON object per line.

## Required fields

| Field | Type | Description |
|-------|------|----------------|
| `name` | string | Entity label (e.g. player name) |
| `team` | string | Short team code or name |
| `position` | string | Position or role |
| `topStats` | array | List of stat objects (see below) |
| `summary` | string | Reference text (used as supervision and as reference for BLEU/ROUGE/BERTScore) |

Each element of `topStats`:

```json
{ "stat": "points", "value": 132.0, "pctl": 99 }
```

- `stat`: name of the statistic (string)
- `value`: number or string the model should ground on
- `pctl`: optional percentile (number); included in the **table** used by PARENT so keep it consistent if you use it in prompts

## Example line

```json
{
  "name": "Alex Example",
  "team": "XYZ",
  "position": "C",
  "topStats": [
    {"stat": "goals", "value": 42, "pctl": 95},
    {"stat": "assists", "value": 30, "pctl": 88}
  ],
  "summary": "Alex Example drives offense with 42 goals and strong assist numbers."
}
```

## Stable IDs (for evaluation)

Predictions are joined to gold using `evaluation.jsonl_table.stable_example_id(row)`, a hash of `name`, `team`, `position`, and `topStats`. Any model you add must emit the **same** `id` for the same gold row. The training CLI (`llm_training/player_summary_advanced.py`, any `--model_preset`) does this automatically when using `--export_predictions` or `--generate_all`.

## Prompt alignment

The training prompt formats stats like:

`stat: value (percentile: pctl)` joined by `; `.

The evaluation **table** for PARENT uses the same convention internally. If you change formatting in training, update `evaluation/jsonl_table.py` so table records stay aligned (see `records_from_example`).

## From your own CSV / database

1. Map each row to one JSON object with the fields above (or see [table-to-text.md](table-to-text.md) for how to generalize the schema and still align training with evaluation).
2. Write JSONL (no trailing comma between lines).
3. Point `--data_path` at your file for train/test and `--gold` at the **same** file for `run_eval` (gold references + table come from each line).

`scripts/generate_top10_stats_jsonl.py` is the **hockey** ETL from CSV to this JSONL shape; other domains follow the same table-to-text steps in [table-to-text.md](table-to-text.md).
