"""Create clean NBA player-game CSVs, with names and separate study splits.

Run from the repository root: uv run python -m scripts.prepare_data --split train
Start small: --season 2024-25 --player-id 2544
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features import FEATURE_COLUMNS, build_features
from scripts.nba_source import NBAData, save_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],
    "validation": ["2025-26"],
    "test": ["2026-27"],
}
SPLIT_FILES = {
    "train": "train_2020_2025.csv",
    "validation": "validation_2025_2026.csv",
    "test": "test_2026_2027.csv",
}


def separate_ready_rows(dataset):
    """Complete-case initial study: no zero filling or learned imputation."""
    numeric = dataset[FEATURE_COLUMNS + ["target_points"]]
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature or target value")
    missing = numeric.isna()
    ready = ~missing.any(axis=1)
    clean = dataset.loc[ready].copy()
    excluded = dataset.loc[~ready].copy()
    excluded["missing_columns"] = ""
    if not excluded.empty:
        excluded["missing_columns"] = missing.loc[~ready].apply(
            lambda row: ",".join(row.index[row]), axis=1,
        )
    return clean, excluded


def prepare_season(season, source, output, player_ids=None, refresh_logs=False):
    """Read this function first: load, calculate, clean, then export."""
    # 1. Load the original NBA records (or reuse cached downloads).
    players = source.games(season, "P", refresh_logs)
    teams = source.games(season, "T", refresh_logs)
    if player_ids:
        players = players[players["PLAYER_ID"].isin(player_ids)]
    if players.empty:
        raise ValueError(f"No appearances for selected players in {season}")
    birthdates = source.birthdates(players["PLAYER_ID"])
    # Full team logs are needed to validate GP and calculate schedule history.
    snapshots = source.team_snapshots(season, teams["GAME_DATE"])

    # 2. Calculate features using only information before each game.
    dataset = build_features(players, teams, birthdates, snapshots)

    # 3. Keep complete rows and explain why other rows were excluded.
    clean, excluded = separate_ready_rows(dataset)
    clean = clean.sort_values(["game_date", "game_id", "player_id"])

    # 4. Save the model data and the exclusion list separately.
    save_csv(clean, output / "seasons" / f"features_{season}.csv")
    save_csv(excluded, output / "excluded" / f"excluded_{season}.csv")
    report = season_report(season, dataset, clean, excluded)
    print(f"{season}: {len(clean):,} clean rows, {len(excluded):,} excluded", flush=True)
    return clean, report


def season_report(season, dataset, clean, excluded):
    return {
        "season": season,
        "source_rows": len(dataset),
        "clean_rows": len(clean),
        "excluded_rows": len(excluded),
        "last_game_date": str(pd.to_datetime(dataset["game_date"]).max().date()),
        "players_in_clean_data": int(clean["player_id"].nunique()),
        "missing_feature_counts": {key: int(value) for key, value in dataset[FEATURE_COLUMNS].isna().sum().items()},
    }


def run(args):
    raw = args.data_dir / "dirty"
    output = args.data_dir / "clean"
    # A small player sample must never overwrite full-study exports.
    if args.player_id:
        selection = "players_" + "_".join(map(str, sorted(set(args.player_id))))
        output = output / "samples" / selection
    source = NBAData(raw, args.offline, args.timeout, args.attempts)
    if args.season:
        chosen = {}
        seasons = [args.season]
    else:
        chosen = SPLITS if args.split == "all" else {args.split: SPLITS[args.split]}
        seasons = []
        for split_seasons in chosen.values():
            seasons.extend(split_seasons)
    frames, reports = {}, []
    for season in seasons:
        frame, report = prepare_season(season, source, output, args.player_id, args.refresh_logs)
        frames[season] = frame
        reports.append(report)
    for split, split_seasons in chosen.items():
        combined = pd.concat([frames[season] for season in split_seasons], ignore_index=True)
        if combined.empty:
            raise ValueError(f"No model-ready rows for {split}; inspect exclusions")
        if combined.duplicated(["game_id", "player_id"]).any():
            raise ValueError(f"Duplicate player-game records in {split}")
        save_csv(combined, output / SPLIT_FILES[split])
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "target_points",
        "split_seasons": chosen,
        "player_ids": args.player_id,
        "missing_value_policy": "Exclude incomplete rows; record exclusions. No learned imputation.",
        "seasons": reports,
    }
    report_name = args.season or args.split
    (output / f"report_{report_name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exports saved in {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split", choices=[*SPLITS, "all"], default="train")
    group.add_argument("--season", help="Process one season without creating combined split files")
    parser.add_argument("--player-id", type=int, action="append", help="Optional smoke-test subset; repeat for multiple players")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--offline", action="store_true", help="Require existing source caches")
    parser.add_argument("--refresh-logs", action="store_true", help="Refresh player/team logs for an ongoing season")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args(argv)
    if args.season:
        if not re.fullmatch(r"\d{4}-\d{2}", args.season):
            parser.error("Season must look like 2025-26")
        if int(args.season[-2:]) != (int(args.season[:4]) + 1) % 100:
            parser.error("Season must span consecutive years")
    if args.timeout < 1 or args.attempts < 1:
        parser.error("timeout and attempts must be positive")
    if args.offline and args.refresh_logs:
        parser.error("offline cannot be combined with refresh-logs")
    return args


if __name__ == "__main__":
    run(parse_args())
