"""Evaluate predictions against gold JSONL (PARENT + optional n-gram / BERT metrics)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from evaluation.automatic_metrics import numeric_stat_coverage, run_optional_metrics
from evaluation.jsonl_table import (
    record_strings_for_lcs,
    records_from_example,
    stable_example_id,
    table_lexicon_from_records,
    tokenize,
)
from evaluation.parent_metric import parent_score


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pred_map(preds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for p in preds:
        eid = p.get("id")
        if not eid:
            raise ValueError("Each prediction line must include string field 'id' (use evaluation.stable_example_id on gold).")
        m[str(eid)] = p
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Table-to-text eval: PARENT, BLEU, ROUGE, chrF++, BERTScore, efficiency.")
    ap.add_argument("--gold", required=True, type=Path, help="Gold JSONL: name, team, position, topStats, summary")
    ap.add_argument("--pred", required=True, type=Path, help="Predictions JSONL: id, generated, optional timing fields")
    ap.add_argument("--out", required=True, type=Path, help="Output JSON report path")
    ap.add_argument(
        "--metrics",
        default="parent,bleu,chrf,rouge,numeric_coverage",
        help="Comma-separated: parent, bleu, chrf, rouge, bertscore, numeric_coverage",
    )
    ap.add_argument(
        "--lambda",
        dest="lambda_fixed",
        type=float,
        default=None,
        help="Optional fixed PARENT lambda (default: per-instance heuristic from paper)",
    )
    ap.add_argument("--bertscore-lang", default="en", help="BERTScore language code")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_map = build_pred_map(load_jsonl(args.pred))
    names = [x.strip().lower() for x in args.metrics.split(",") if x.strip()]

    per_example: List[Dict[str, Any]] = []
    refs: List[str] = []
    hyps: List[str] = []

    for ex in gold_rows:
        eid = stable_example_id(ex)
        p = pred_map.get(eid)
        if p is None:
            raise KeyError(
                f"No prediction for id={eid!r}. Ensure predictions use evaluation.stable_example_id(gold_row)."
            )

        ref = ex.get("summary") or ""
        gen = p.get("generated") or ""
        refs.append(ref)
        hyps.append(gen)

        records = records_from_example(ex)
        tlex = table_lexicon_from_records(records)
        rtoks = record_strings_for_lcs(records)
        gt = tokenize(gen)
        rt = tokenize(ref)

        row: Dict[str, Any] = {"id": eid}

        if "numeric_coverage" in names:
            row["numeric_coverage"] = numeric_stat_coverage(gen, records)

        if "parent" in names:
            row.update(parent_score(gt, rt, tlex, rtoks, lambda_fixed=args.lambda_fixed))

        eff: Dict[str, Any] = {}
        if p.get("generation_time_s") is not None:
            eff["generation_time_s"] = float(p["generation_time_s"])
        if p.get("num_output_tokens") is not None:
            eff["num_output_tokens"] = float(p["num_output_tokens"])
        if p.get("peak_gpu_mb") is not None:
            eff["peak_gpu_mb"] = float(p["peak_gpu_mb"])
        if eff.get("generation_time_s") and eff["generation_time_s"] > 0 and "num_output_tokens" in eff:
            eff["tokens_per_sec"] = eff["num_output_tokens"] / eff["generation_time_s"]
        if eff:
            row["efficiency"] = eff

        per_example.append(row)

    report: Dict[str, Any] = {
        "metrics_requested": names,
        "num_examples": len(per_example),
        "per_example": per_example,
        "automatic": {},
        "efficiency": {},
    }

    opt = [m for m in names if m in ("bleu", "chrf", "rouge", "bertscore")]
    if opt:
        report["automatic"].update(run_optional_metrics(refs, hyps, opt, bertscore_lang=args.bertscore_lang))

    if "numeric_coverage" in names:
        report["automatic"]["numeric_coverage_mean"] = statistics.mean(
            r["numeric_coverage"] for r in per_example if "numeric_coverage" in r
        )

    if "parent" in names:
        report["automatic"]["parent_f1_mean"] = statistics.mean(r["parent_f1"] for r in per_example)

    times = [
        r["efficiency"]["generation_time_s"]
        for r in per_example
        if r.get("efficiency") and "generation_time_s" in r["efficiency"]
    ]
    tps = [
        r["efficiency"]["tokens_per_sec"]
        for r in per_example
        if r.get("efficiency") and "tokens_per_sec" in r["efficiency"]
    ]
    if times:
        report["efficiency"]["generation_time_s"] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
        }
    if tps:
        report["efficiency"]["tokens_per_sec"] = {
            "mean": statistics.mean(tps),
            "stdev": statistics.stdev(tps) if len(tps) > 1 else 0.0,
        }

    peaks = [
        r["efficiency"]["peak_gpu_mb"]
        for r in per_example
        if r.get("efficiency") and "peak_gpu_mb" in r["efficiency"]
    ]
    if peaks:
        report["efficiency"]["peak_gpu_mb"] = {
            "mean": statistics.mean(peaks),
            "max": max(peaks),
            "stdev": statistics.stdev(peaks) if len(peaks) > 1 else 0.0,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "per_example"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()