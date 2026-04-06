"""
PARENT-style metric (Dhingra et al., 2019): Precision And Recall of Entailed N-grams
from the Table — https://arxiv.org/abs/1906.01081

This file implements the **word-overlap** entailment model for w(g) (§3 of the paper).
External implementations (spike notes for reproducibility / parity checks):
- **google-research/language** — `language/table_text_eval` (TensorFlow-era; clone and run in a
  matching env if you need exact paper numbers).
- **KaijuML/parent** — `pip install git+https://github.com/KaijuML/parent.git` (faster port);
  may fail to build on some Windows / Python 3.12 setups (setuptools/pkg_resources).

When those are unavailable, this in-repo implementation matches the published equations
for the word-overlap variant.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

EPS = 1e-5
N_MAX = 4


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def ngram_counts(tokens: List[str], n: int) -> Counter:
    c: Counter = Counter()
    if len(tokens) < n:
        return c
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i : i + n])
        c[g] += 1
    return c


def word_overlap_w(g: Tuple[str, ...], table_lex: set) -> float:
    if not g:
        return EPS
    hits = sum(1 for t in g if t in table_lex)
    return hits / len(g)


def entailed_precision_n(gen: List[str], ref: List[str], table_lex: set, n: int) -> float:
    cg = ngram_counts(gen, n)
    cr = ngram_counts(ref, n)
    if not cg:
        return EPS
    num = 0.0
    den = 0.0
    for g, cnt in cg.items():
        w = word_overlap_w(g, table_lex)
        cgr = min(cnt, cr.get(g, 0))
        num += cnt * w + cgr * (1.0 - w)
        den += cnt
    return max(num / den, EPS)


def entailed_recall_R_n(gen: List[str], ref: List[str], table_lex: set, n: int) -> float:
    cg = ngram_counts(gen, n)
    cr = ngram_counts(ref, n)
    if not cr:
        return EPS
    num = 0.0
    den = 0.0
    for g, cnt_r in cr.items():
        w = word_overlap_w(g, table_lex)
        cgr = min(cg.get(g, 0), cnt_r)
        num += cgr * w
        den += cnt_r * w
    if den <= 0:
        return EPS
    return max(num / den, EPS)


def table_recall(gen: List[str], record_token_seqs: List[List[str]]) -> float:
    if not record_token_seqs:
        return EPS
    total = 0.0
    for toks in record_token_seqs:
        ln = max(len(toks), 1)
        total += lcs_length(toks, gen) / ln
    return max(total / len(record_token_seqs), EPS)


def geom_mean(vals: List[float]) -> float:
    vals = [max(v, EPS) for v in vals]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def reference_table_recall_lambda(ref: List[str], record_token_seqs: List[List[str]]) -> float:
    """Paper heuristic: 1 - lambda = table recall of reference text (Eq. 6 on R)."""
    er_t_ref = table_recall(ref, record_token_seqs)
    er_t_ref = min(max(er_t_ref, 0.0), 1.0)
    return 1.0 - er_t_ref


def parent_score(
    gen_tokens: List[str],
    ref_tokens: List[str],
    table_lex: set,
    record_token_seqs: List[List[str]],
    lambda_fixed: float | None = None,
) -> dict:
    eparts = [entailed_precision_n(gen_tokens, ref_tokens, table_lex, n) for n in range(1, N_MAX + 1)]
    ep = geom_mean(eparts)

    err_parts = [entailed_recall_R_n(gen_tokens, ref_tokens, table_lex, n) for n in range(1, N_MAX + 1)]
    er_r = geom_mean(err_parts)

    er_t = table_recall(gen_tokens, record_token_seqs)

    if lambda_fixed is not None:
        lam = lambda_fixed
    else:
        lam = reference_table_recall_lambda(ref_tokens, record_token_seqs)

    log_er = (1.0 - lam) * math.log(max(er_r, EPS)) + lam * math.log(max(er_t, EPS))
    er = math.exp(log_er)
    er = max(er, EPS)

    if ep + er <= 0:
        f1 = EPS
    else:
        f1 = 2.0 * ep * er / (ep + er)

    return {
        "parent_f1": f1,
        "parent_precision": ep,
        "parent_recall": er,
        "parent_recall_ref": er_r,
        "parent_recall_table": er_t,
        "parent_lambda": lam,
    }