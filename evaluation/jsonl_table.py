"""Map gold JSONL rows to table records and stable IDs (aligned with training prompts)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

Record = Tuple[str, str]


def stable_example_id(example: Dict[str, Any]) -> str:
    """Content-based ID so predictions can join to gold without row order."""
    payload = {
        "name": example["name"],
        "team": example["team"],
        "position": example["position"],
        "topStats": example["topStats"],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{example['name']}|{example['team']}|{h}"


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens (keeps decimals in numbers)."""
    if not text:
        return []
    return re.findall(r"[a-z0-9.]+", text.lower()) or []


def records_from_example(example: Dict[str, Any]) -> List[Record]:
    """(attribute, value) records for PARENT; matches training stat formatting."""
    rows: List[Record] = [
        ("name", str(example["name"]).strip()),
        ("team", str(example["team"]).strip()),
        ("position", str(example["position"]).strip()),
    ]
    for s in example.get("topStats") or []:
        stat = str(s.get("stat", "")).strip()
        val = s.get("value")
        pctl = s.get("pctl")
        if stat:
            value_str = f"{val} (percentile: {pctl})" if pctl is not None else str(val)
            rows.append((stat, value_str.strip()))
    return rows


def table_lexicon_from_records(records: List[Record]) -> set:
    """Tokens in attribute names and values (word-overlap entailment)."""
    lex = set()
    for attr, val in records:
        lex.update(tokenize(attr))
        lex.update(tokenize(val))
    return lex


def record_strings_for_lcs(records: List[Record]) -> List[List[str]]:
    """Per-record tokens for table recall LCS."""
    out = []
    for attr, val in records:
        out.append(tokenize(f"{attr} {val}"))
    return out