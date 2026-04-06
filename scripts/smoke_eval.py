#!/usr/bin/env python3
"""CI / local smoke test: toy gold+pred JSONL and evaluation.run_eval (no GPU, no models)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    gold = {
        "name": "Smoke Player",
        "team": "SMK",
        "position": "C",
        "topStats": [{"stat": "points", "value": 50.0, "pctl": 90}],
        "summary": "Smoke Player scores 50 points for SMK.",
    }
    sys.path.insert(0, str(root))
    from evaluation.jsonl_table import stable_example_id

    eid = stable_example_id(gold)
    pred = {
        "id": eid,
        "generated": "Smoke Player had 50 points this season.",
        "generation_time_s": 0.1,
        "num_output_tokens": 8.0,
    }

    tmp = root / "outputs" / "_smoke_eval"
    tmp.mkdir(parents=True, exist_ok=True)
    gold_path = tmp / "gold.jsonl"
    pred_path = tmp / "pred.jsonl"
    report_path = tmp / "report.json"

    gold_path.write_text(json.dumps(gold) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "evaluation.run_eval",
        "--gold",
        str(gold_path),
        "--pred",
        str(pred_path),
        "--out",
        str(report_path),
        "--metrics",
        "parent,numeric_coverage",
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    if r.returncode != 0:
        return r.returncode
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "per_example" in data and len(data["per_example"]) == 1
    print("smoke_eval OK:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
