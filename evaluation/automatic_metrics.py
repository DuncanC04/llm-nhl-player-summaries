"""BLEU, ROUGE, chrF++, BERTScore, and numeric coverage — optional dependencies."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


def numeric_stat_coverage(generated: str, records: List[Tuple[str, str]]) -> float:
    """
    Share of numeric literals in table values found in generated text (substring).
    Heuristic regression signal only (not a replacement for PARENT).
    """
    g = generated.lower()
    hits = 0
    total = 0
    for _attr, val in records:
        for m in re.finditer(r"\d+\.?\d*", str(val).lower()):
            num = m.group(0)
            total += 1
            if num in g:
                hits += 1
            elif num.endswith(".0") and num[:-2] in g:
                hits += 1
            elif "." in num and num.rstrip("0").rstrip(".") in g:
                hits += 1
    if total == 0:
        return 1.0
    return hits / total


def compute_rouge(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {k: 0.0 for k in ["rouge1", "rouge2", "rougeL"]}
    n = 0
    for ref, hyp in zip(refs, hyps):
        if not hyp.strip():
            continue
        scores = scorer.score(ref, hyp)
        for k in agg:
            agg[k] += scores[k].fmeasure
        n += 1
    if n == 0:
        return {k: 0.0 for k in agg}
    return {k: v / n for k, v in agg.items()}


def compute_bertscore(refs: List[str], hyps: List[str], lang: str = "en") -> Dict[str, float]:
    import bert_score

    p, r, f1 = bert_score.score(hyps, refs, lang=lang, rescale_with_baseline=False)
    return {
        "bertscore_precision": float(p.mean()),
        "bertscore_recall": float(r.mean()),
        "bertscore_f1": float(f1.mean()),
    }


def run_optional_metrics(
    refs: List[str],
    hyps: List[str],
    metric_names: List[str],
    bertscore_lang: str = "en",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    names = {m.strip().lower() for m in metric_names}
    if "bleu" in names or "chrf" in names:
        from sacrebleu.metrics import BLEU, CHRF

        if "bleu" in names:
            out["bleu"] = float(BLEU().corpus_score(hyps, [refs]).score)
        if "chrf" in names:
            out["chrf_pp"] = float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)
    if "rouge" in names:
        out.update(compute_rouge(refs, hyps))
    if "bertscore" in names:
        out.update(compute_bertscore(refs, hyps, lang=bertscore_lang))
    return out
