"""Prepare completed 2026–27 appearances through the registered RQ3 cutoff."""

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from scripts.experiment_config import SPLITS, SPLIT_FILES
from scripts.features import FEATURE_COLUMNS
from scripts.modeling import TARGET
from scripts.nba_source import NBAData, save_csv
from scripts.prepare_data import PROJECT_ROOT, prepare_season

SEASON, = SPLITS["test"]
CUTOFF = "2026-11-22"
TEST_DIR = PROJECT_ROOT / "data/clean" / f"rq3_through_{CUTOFF}"


class CutoffSource(NBAData):
    """Cache date-bounded game logs separately from full-season logs."""

    def games(self, season, kind, refresh=False):
        label = "players" if kind == "P" else "teams"
        frame = self.table(
            f"{season}/through_{CUTOFF}/{label}.csv",
            lambda: leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation=kind,
                date_to_nullable=pd.Timestamp(CUTOFF).strftime("%m/%d/%Y"),
                timeout=self.timeout,
            ).get_data_frames()[0],
            ["SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ID", "WL", "PTS"],
            refresh=refresh,
        )
        if (frame.empty or not frame["SEASON_ID"].astype(str).eq("22026").all()
                or not frame["WL"].isin(["W", "L"]).all()
                or pd.to_datetime(frame["GAME_DATE"]).max() > pd.Timestamp(CUTOFF)):
            raise ValueError(f"Expected completed regular-season {label} logs through {CUTOFF}")
        return frame.assign(season=season)


def prepare(raw, output, offline):
    source = CutoffSource(raw, offline=offline)
    players = source.games(SEASON, "P")
    teams = source.games(SEASON, "T")
    keys = ["GAME_ID", "TEAM_ID"]
    player_points = players.groupby(keys)["PTS"].sum().sort_index()
    team_points = teams.set_index(keys)["PTS"].sort_index()
    pd.testing.assert_series_equal(player_points, team_points, check_dtype=False)

    # The official cumulative GP through the cutoff also checks for whole games
    # missing from both player and team logs. team_snapshots requests D minus 1.
    next_day = pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)
    closing = source.team_snapshots(SEASON, [next_day]).set_index("TEAM_ID")["GP"]
    games_played = teams.groupby("TEAM_ID").size().reindex(closing.index, fill_value=0)
    if set(teams["TEAM_ID"]) - set(closing.index) or not closing.eq(games_played).all():
        raise ValueError("Game-log coverage disagrees with team GP through the cutoff")

    clean, audit = prepare_season(SEASON, source, output)
    if clean.empty:
        raise ValueError("No eligible appearances in the locked window")
    save_csv(clean, output / SPLIT_FILES["test"])
    report = {"cutoff": CUTOFF, "first_source_game_date": str(pd.to_datetime(teams["GAME_DATE"]).min().date()),
              "source_games": int(teams["GAME_ID"].nunique()), "feature_columns": FEATURE_COLUMNS,
              "target_column": TARGET, "split_seasons": {"test": [SEASON]}, "player_ids": None,
              "seasons": [audit]}
    (output / "report_test.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/dirty")
    parser.add_argument("--output-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(prepare(args.raw_dir, args.output_dir, args.offline))
