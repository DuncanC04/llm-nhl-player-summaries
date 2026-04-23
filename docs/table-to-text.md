# Table to text: general approach and this project

This note is for anyone using the repo as a **reference for table-to-text** (structured fields → natural language), not only for hockey stats.

## What “table to text” means here

You have **structured input** (rows, key–value groups, or a small relational slice) and you want **one or more sentences** that describe, summarize, or explain that structure. The model must **ground** its wording in the numbers and labels you provide.

That pattern is the same whether your “table” is sports stats, a product catalog row, lab results, or a database export.

## A recipe that works for any domain

1. **Fix a canonical record format**  
   Choose one object per training example (e.g. one JSON object per line in JSONL). Every field the model should use must live in that record. The reference text (what humans or a pipeline produced) is your **target**.

2. **Linearize the table for the model**  
   Models read **tokens**, not SQL. Turn each record into a **stable prompt template**: fixed section headers, delimiter-separated stats, or a short bullet list. The only requirement is **consistency**: training, inference, and any metric that reads “the table” should use the **same** rendering rules.

3. **Pair input with supervision**  
   Each example is `(rendered_structure, reference_text)`. For a causal LM, common practice is **prompt + completion** in one string (SFT), or a **chat template** with a user message (table) and an assistant message (summary). This repo uses plain continuation for the **`mistral`** preset and a chat template for **`phi-3-mini`** (see `llm_training/player_summary/presets/`).

4. **Split and train**  
   Hold out a validation split. Rows are **shuffled** before splitting by default (configurable seed) so validation is not biased by JSONL order. Fine-tune with **QLoRA** (default base: **Mistral-7B**; optional **Phi-3-mini** preset).

5. **Align evaluation with the prompt**  
   If you use **structure-aware** scores (e.g. PARENT, which compares generated text to a set of `(attribute, value)` records), build those records with the **same** logic as the training prompt. If you change how stats appear in the prompt, update the code that builds the metric’s table or the scores will not measure what you think.

6. **Stable IDs for joining predictions to gold**  
   Export predictions with an `id` per example that depends on **content**, not line order, so you can merge with references and human ratings after shuffling or re-running pipelines.

7. **Measure what you care about**  
   Combine **n-gram / overlap** metrics (BLEU, ROUGE, chrF), **semantic** similarity (e.g. BERTScore) if useful, **numeric grounding** checks where applicable, **table-aware** metrics where implemented, and **human** rubrics for fluency and faithfulness.

## How this repository implements that

| Step | What we did |
|------|----------------|
| Canonical format | JSONL lines with `name`, `team`, `position`, `topStats[]`, and `summary` (see [custom-dataset.md](custom-dataset.md)). |
| Linearization | `format_stats_text` + `create_prompt` in `llm_training/player_summary/prompts.py`: each stat becomes `stat: value (percentile: pctl)` joined by `; `. |
| Supervision | `data_pipeline.create_training_example` delegates to the active **preset**: plain `prompt + summary` for `mistral`, or `tokenizer.apply_chat_template` (user/assistant) for `phi-3-mini`. |
| Training | `llm_training/player_summary/`: 4-bit load, LoRA targets resolved per preset, `SFTTrainer` / `SFTConfig`; CLI via `player_summary_advanced.py` (see [models-and-evaluation.md](models-and-evaluation.md)). |
| Data from raw tables | `scripts/generate_top10_stats_jsonl.py` maps hockey CSV rows into the JSONL schema; it is one **instance** of step 1–2 for this domain. |
| Eval table + IDs | `evaluation/jsonl_table.py`: `records_from_example` builds PARENT’s `(attribute, value)` rows to match the stat string format above; `stable_example_id` hashes the structured fields used in the prompt. |
| Metrics + human study | `python -m evaluation.run_eval` on gold JSONL + prediction JSONL; optional merge with `evaluation/merge_human` and the rubric templates. |

## Adapting to another table-to-text task

You do **not** need the hockey field names. For a new task:

1. Define your JSONL schema (which columns are “the table,” which field is the reference `summary` or equivalent).
2. Implement **one** prompt builder (like `create_prompt`) that always renders those fields the same way.
3. If you keep PARENT (or any code that uses `records_from_example`), extend or replace `evaluation/jsonl_table.py` so **records** match what the model saw in the prompt, and adjust `stable_example_id` to hash the fields that uniquely identify each example.
4. Point `--data_path` at your JSONL and train as in [getting-started.md](getting-started.md).

The hockey CSV script is an **example ETL**; the reusable pattern is the JSONL contract, the prompt linearization in `player_summary/prompts.py`, the modular training package under `llm_training/player_summary/`, and the evaluation alignment in `jsonl_table.py`.
