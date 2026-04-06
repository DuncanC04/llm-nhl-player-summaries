"""Join human rubric CSV rows with eval_report.json for analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


LIKERT_COLS = ("faithfulness", "fluency", "informativeness", "overall")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge human ratings with automatic eval JSON")
    ap.add_argument("--human", type=Path, required=True, help="CSV from human_rubric_template.csv")
    ap.add_argument("--report", type=Path, required=True, help="eval_report.json from run_eval.py")
    ap.add_argument("--out", type=Path, required=True, help="Output merged JSON")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = load_csv(args.human)
    by_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        eid = (row.get("example_id") or row.get("id") or "").strip()
        if eid:
            by_id[eid].append(row)

    human_per_example: Dict[str, Any] = {}
    for eid, lst in by_id.items():
        agg: Dict[str, float] = {}
        for col in LIKERT_COLS:
            vals: List[float] = []
            for r in lst:
                v = r.get(col)
                if v is not None and str(v).strip() != "":
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            if vals:
                agg[f"{col}_mean"] = mean(vals)
                agg[f"{col}_n_raters"] = float(len(vals))
        human_per_example[eid] = agg

    automatic_by_id: Dict[str, Any] = {}
    for row in report.get("per_example") or []:
        eid = row.get("id")
        if eid:
            automatic_by_id[str(eid)] = {k: v for k, v in row.items() if k != "id"}

    combined_examples: Dict[str, Any] = {}
    for eid in set(human_per_example) | set(automatic_by_id):
        combined_examples[eid] = {
            "human": human_per_example.get(eid, {}),
            "automatic": automatic_by_id.get(eid, {}),
        }

    merged = {
        "automatic_report": report,
        "human_per_example": human_per_example,
        "automatic_by_id": automatic_by_id,
        "combined_by_id": combined_examples,
        "num_rater_rows": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(human_per_example)} example ids with human rows)")


if __name__ == "__main__":
    main()