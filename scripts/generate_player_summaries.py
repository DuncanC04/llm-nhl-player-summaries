import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def read_skaters(csv_path: Path, min_games: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Use only situation == 'all' rows to avoid duplication
    df = df[df["situation"].astype(str).str.lower() == "all"].copy()
    # Minimum sample filter
    df = df[df["games_played"].fillna(0) >= min_games].copy()

    # Position grouping: Forwards vs Defense
    def map_pos(p: str) -> str:
        p = str(p).upper()
        return "D" if p == "D" else "F"

    df["posGroup"] = df["position"].apply(map_pos)

    # Ensure time exists
    df = df.dropna(subset=["icetime"])  # ensure time exists
    return df


def read_goalies(csv_path: Path, min_games: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["situation"].astype(str).str.lower() == "all"].copy()
    df = df[df["games_played"].fillna(0) >= min_games].copy()
    df = df.dropna(subset=["icetime"])  # ensure time exists
    return df


def add_percentiles(df: pd.DataFrame, cols: List[str], group_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()

    def pct_rank(s: pd.Series) -> pd.Series:
        # Rank method=average, pct=True gives 0..1 inclusive
        return s.rank(pct=True, method="average")

    if group_col is None:
        for c in cols:
            out[f"{c}_pctl"] = pct_rank(out[c].astype(float))
        return out

    # Groupwise percentiles
    for c in cols:
        out[f"{c}_pctl"] = (
            out.groupby(group_col)[c].transform(lambda s: pct_rank(s.astype(float)))
        )
    return out


def get_numeric_stat_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in set(exclude)]


def build_extreme_highlights(
    row: pd.Series,
    cols: List[str],
    high_thresh: float,
    low_thresh: float,
) -> List[Tuple[str, float, float]]:
    highlights: List[Tuple[str, float, float]] = []  # (col, value, percentile)
    for c in cols:
        p = row.get(f"{c}_pctl", np.nan)
        if pd.isna(p):
            continue
        if p >= high_thresh or p <= low_thresh:
            v = row.get(c, np.nan)
            if pd.notna(v):
                highlights.append((c, float(v), float(p)))
    # sort by distance from median (extremeness)
    highlights.sort(key=lambda t: max(t[2], 1 - t[2]), reverse=True)
    return highlights


def format_percentile(p: float) -> str:
    pct = int(round(p * 100))
    return f"{pct}th percentile"


def round_num(x: float, ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{ndigits}f}"


def summarize_with_extremes(
    row: pd.Series,
    highlights: List[Tuple[str, float, float]],
    role_label: str,
    max_highlights: int,
) -> str:
    name = row.get("name", "Unknown")
    team = row.get("team", "-")
    pos = row.get("position", None)
    games = int(row.get("games_played", 0))

    if not highlights:
        if pos is None:
            return f"{name} ({team}, {role_label}) played {games} games."
        return f"{name} ({team}, {pos}) played {games} games."

    use = highlights[:max_highlights]
    bits = []
    for col, val, p in use:
        direction = "top" if p >= 0.5 else "bottom"
        bits.append(f"{col}: {round_num(val)} ({direction} {format_percentile(p)})")

    if pos is None:
        head = f"{name} ({team}, {role_label}) had {len(highlights)} extreme stat(s) over {games} games."
    else:
        head = f"{name} ({team}, {pos}) had {len(highlights)} extreme stat(s) over {games} games."
    return " ".join([head, "; ".join(bits) + "."])


def generate_records_for_group(
    df: pd.DataFrame,
    group_col: Optional[str],
    high_thresh: float,
    low_thresh: float,
    id_exclude: List[str],
    max_highlights: int,
    role_label: str,
    record_cap: Optional[int] = None,
) -> List[dict]:
    cols = get_numeric_stat_columns(df, exclude=id_exclude)
    dfp = add_percentiles(df, cols, group_col=group_col)
    records: List[dict] = []
    for _, r in dfp.iterrows():
        highlights = build_extreme_highlights(r, cols, high_thresh, low_thresh)
        if not highlights:
            continue
        completion = summarize_with_extremes(r, highlights, role_label, max_highlights)
        prompt = f"Player: {r['name']}"
        records.append({"prompt": prompt, "completion": completion})
        if record_cap is not None and len(records) >= record_cap:
            break
    return records


def write_jsonl(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL of player summaries highlighting top/bottom percentile stats across ALL numeric columns."
    )
    parser.add_argument(
        "--skaters",
        type=Path,
        default=Path("Data/skaters_24_25.csv"),
        help="Path to skaters CSV",
    )
    parser.add_argument(
        "--goalies",
        type=Path,
        default=Path("Data/goalies_24_25.csv"),
        help="Path to goalies CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Data/out/standout_players.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--min-games", type=int, default=15, help="Minimum games played filter")
    parser.add_argument("--high-pctl", type=float, default=0.95, help="High percentile threshold")
    parser.add_argument("--low-pctl", type=float, default=0.05, help="Low percentile threshold")
    parser.add_argument("--max-highlights", type=int, default=6, help="Max highlights per player in summary")
    parser.add_argument("--cap-skaters", type=int, default=None, nargs="?", help="Cap number of skater records (optional)")
    parser.add_argument("--cap-goalies", type=int, default=None, nargs="?", help="Cap number of goalie records (optional)")
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for tie-breaking/shuffling",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load
    skaters_df = read_skaters(args.skaters, min_games=args.min_games)
    goalies_df = read_goalies(args.goalies, min_games=args.min_games)

    # ID/administrative columns to exclude from stat scans
    skater_exclude = [
        "playerId",
        "season",
        # keep games_played as a stat
    ]
    # Keep icetime as a stat; do not exclude

    goalie_exclude = [
        "playerId",
        "season",
        # keep games_played as a stat
    ]

    # Generate records
    records_skaters = generate_records_for_group(
        df=skaters_df,
        group_col="posGroup",
        high_thresh=args.high_pctl,
        low_thresh=args.low_pctl,
        id_exclude=skater_exclude,
        max_highlights=args.max_highlights,
        role_label="Skater",
        record_cap=args.cap_skaters,
    )

    records_goalies = generate_records_for_group(
        df=goalies_df,
        group_col=None,
        high_thresh=args.high_pctl,
        low_thresh=args.low_pctl,
        id_exclude=goalie_exclude,
        max_highlights=args.max_highlights,
        role_label="G",
        record_cap=args.cap_goalies,
    )

    # Combine and shuffle
    records: List[dict] = records_skaters + records_goalies

    # Shuffle to mix positions
    rng = np.random.default_rng(args.seed)
    rng.shuffle(records)

    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()


