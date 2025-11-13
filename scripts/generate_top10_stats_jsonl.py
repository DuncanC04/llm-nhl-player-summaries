import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Explicit allowlist of columns to keep in the working DataFrames (and to select stats from)
COLUMNS_TO_KEEP: List[str] = [
    # Core Player & Game Context
    "playerId",
    "season",
    "name",
    "team",
    "position",
    "situation",

    # Individual Offensive Production (I_F) - Actual and Expected
    "I_F_shotsOnGoal",
    "I_F_missedShots",
    "I_F_blockedShotAttempts",
    "I_F_shotAttempts",
    "I_F_goals",
    "I_F_primaryAssists",
    "I_F_secondaryAssists",
    "I_F_points",
    "I_F_flurryScoreVenueAdjustedxGoals",
    # Shot danger profile
    "I_F_lowDangerShots",
    "I_F_mediumDangerShots",
    "I_F_highDangerShots",
    "I_F_rebounds",
    "I_F_reboundGoals",

    # Individual Defense & Playmaking
    "I_F_takeaways",
    "I_F_giveaways",
    "penalties",
    "penaltiesDrawn",
    "I_F_hits",
    "shotsBlockedByPlayer",

    # On-Ice & Off-Ice Impact (Percentages)
    "onIce_xGoalsPercentage",
    "onIce_corsiPercentage",
    "onIce_fenwickPercentage",
    "offIce_xGoalsPercentage",
    "offIce_corsiPercentage",
    "offIce_fenwickPercentage",

    # On-Ice Team Results (Actual Goals and Expected Goals)
    "OnIce_F_goals",
    "OnIce_A_goals",
    "OnIce_F_flurryScoreVenueAdjustedxGoals",
    "OnIce_A_flurryScoreVenueAdjustedxGoals",
    "timeOnBench",
]

CORE_CONTEXT_COLUMNS: List[str] = [
    "playerId",
    "season",
    "name",
    "team",
    "position",
    "situation",
]

def read_skaters(csv_path: Path, min_games: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["situation"].astype(str).str.lower() == "all"].copy()
    df = df[df["games_played"].fillna(0) >= min_games].copy()

    def map_pos(p: str) -> str:
        p = str(p).upper()
        return "D" if p == "D" else "F"

    df["posGroup"] = df["position"].apply(map_pos)
    df = df.dropna(subset=["icetime"])  # ensure time exists
    # Restrict to allowed columns if present
    keep = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    extra_context = ["posGroup"]
    for c in extra_context:
        if c not in keep and c in df.columns:
            keep.append(c)
    if keep:
        df = df[keep].copy()
    return df


def read_goalies(csv_path: Path, min_games: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["situation"].astype(str).str.lower() == "all"].copy()
    df = df[df["games_played"].fillna(0) >= min_games].copy()
    df = df.dropna(subset=["icetime"])  # ensure time exists
    # Restrict to allowed columns if present
    keep = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    if keep:
        df = df[keep].copy()
    return df


def add_percentiles(
    df: pd.DataFrame, cols: List[str], group_col: Optional[str] = None
) -> pd.DataFrame:
    out = df.copy()

    def pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True, method="average")

    if group_col is None:
        for c in cols:
            out[f"{c}_pctl"] = pct_rank(out[c].astype(float))
        return out

    for c in cols:
        out[f"{c}_pctl"] = (
            out.groupby(group_col)[c].transform(lambda s: pct_rank(s.astype(float)))
        )
    return out


def get_numeric_stat_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_set = set(exclude)
    return [c for c in numeric_cols if c not in exclude_set]


def collect_top_stats_for_row(
    row: pd.Series, cols: List[str], top_thresh: float
) -> List[Tuple[str, float, float]]:
    top_stats: List[Tuple[str, float, float]] = []  # (col, value, percentile)
    for c in cols:
        p = row.get(f"{c}_pctl", np.nan)
        if pd.isna(p) or p < top_thresh:
            continue
        v = row.get(c, np.nan)
        if pd.notna(v):
            top_stats.append((c, float(v), float(p)))
    # Sort by percentile descending, then by value descending
    top_stats.sort(key=lambda t: (t[2], t[1]), reverse=True)
    return top_stats


def write_jsonl(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate JSONL with per-player top-percentile stats only (no low stats)."
        )
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
        default=Path("Data/out/aiTop10Stats.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--min-games", type=int, default=15, help="Minimum games played filter"
    )
    parser.add_argument(
        "--top-pctl",
        type=float,
        default=0.90,
        help="Top percentile threshold (e.g., 0.90 for top 10%)",
    )
    parser.add_argument(
        "--include-cols",
        type=str,
        default="",
        help=(
            "Comma-separated list of stat columns to include (case-insensitive). If empty, uses built-in allowlist."
        ),
    )
    parser.add_argument(
        "--include-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file with one column name per line to include. Overrides built-in allowlist."
        ),
    )
    parser.add_argument(
        "--cap-skaters",
        type=int,
        default=None,
        nargs="?",
        help="Cap number of skater records (optional)",
    )
    parser.add_argument(
        "--cap-goalies",
        type=int,
        default=None,
        nargs="?",
        help="Cap number of goalie records (optional)",
    )

    args = parser.parse_args()

    # Parse include columns. If none provided, default to built-in allowlist.
    include_cols: List[str] = []
    if args.include_cols:
        include_cols.extend([c.strip() for c in args.include_cols.split(",") if c.strip()])
    if args.include_file is not None and args.include_file.exists():
        with args.include_file.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    include_cols.append(name)
    if not include_cols:
        # Default to only stat columns, not core context
        include_cols = [
            c for c in COLUMNS_TO_KEEP if c not in set(CORE_CONTEXT_COLUMNS)
        ]

    # Load
    skaters_df = read_skaters(args.skaters, min_games=args.min_games)
    goalies_df = read_goalies(args.goalies, min_games=args.min_games)

    # Resolve included numeric columns
    def resolve_included_numeric_columns(df: pd.DataFrame, include_names: List[str]) -> List[str]:
        if not include_names:
            return []
        name_map = {c.lower(): c for c in df.columns.tolist()}
        resolved: List[str] = []
        for name in include_names:
            key = str(name).strip().lower()
            if not key:
                continue
            if key in name_map:
                resolved.append(name_map[key])
        numeric_set = set(df.select_dtypes(include=[np.number]).columns.tolist())
        return [c for c in resolved if c in numeric_set]

    skater_stat_cols = resolve_included_numeric_columns(skaters_df, include_cols)
    goalie_stat_cols = resolve_included_numeric_columns(goalies_df, include_cols)

    if not skater_stat_cols and not goalie_stat_cols:
        raise SystemExit(
            "No included columns matched numeric columns in either dataset. Provide --include-cols or --include-file with valid column names."
        )

    skaters_with_pctl = add_percentiles(
        skaters_df, skater_stat_cols, group_col="posGroup"
    )
    goalies_with_pctl = add_percentiles(
        goalies_df, goalie_stat_cols, group_col=None
    )

    records: List[dict] = []

    # Skaters
    for _, r in skaters_with_pctl.iterrows():
        top_stats = collect_top_stats_for_row(r, skater_stat_cols, args.top_pctl)
        if not top_stats:
            continue
        name = r.get("name", "Unknown")
        team = r.get("team", "")
        position = r.get("position", "")
        top_stats_out = [
            {"stat": c, "value": v, "pctl": round(p * 100)} for (c, v, p) in top_stats
        ]
        records.append({
            "name": name,
            "team": team,
            "position": position,
            "topStats": top_stats_out,
            "summary": ""
        })
        if args.cap_skaters is not None and len(records) >= args.cap_skaters:
            break

    # Goalies
    goalie_count_added = 0
    for _, r in goalies_with_pctl.iterrows():
        top_stats = collect_top_stats_for_row(r, goalie_stat_cols, args.top_pctl)
        if not top_stats:
            continue
        name = r.get("name", "Unknown")
        team = r.get("team", "")
        position = r.get("position", "")
        top_stats_out = [
            {"stat": c, "value": v, "pctl": round(p * 100)} for (c, v, p) in top_stats
        ]
        records.append({
            "name": name,
            "team": team,
            "position": position,
            "topStats": top_stats_out,
            "summary": ""
        })
        goalie_count_added += 1
        if args.cap_goalies is not None and goalie_count_added >= args.cap_goalies:
            break

    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()


