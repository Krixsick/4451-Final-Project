"""Synthetic-data checks; these do not contact the NBA."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.features import FEATURE_COLUMNS, build_features
from scripts.nba_source import NBAData, save_csv
from scripts.prepare_data import parse_args, run, separate_ready_rows


def fixture(season="2024-25"):
    players, teams, snapshots = [], [], []
    year = int(season[:4]) + 1
    for day in range(1, 13):
        date = pd.Timestamp(year, 1, day)
        shared = {"season": season, "GAME_ID": f"002{year}{day:03}", "GAME_DATE": date}
        for team in [1, 2]:
            teams.append({**shared, "TEAM_ID": team, "MATCHUP": "A vs. B" if team == 1 else "B @ A"})
            snapshots.append({"season": season, "GAME_DATE": date, "TEAM_ID": team,
                              "GP": day - 1, "PACE": 100 if team == 1 else 90,
                              "DEF_RATING": 112 if team == 1 else 108,
                              "STATS_THROUGH": date - pd.Timedelta(days=1)})
        if day != 5:
            players.append({**shared, "PLAYER_ID": 10, "PLAYER_NAME": "Synthetic José",
                            "TEAM_ID": 1, "TEAM_ABBREVIATION": "AAA", "PTS": day,
                            "MIN": 30, "FGA": 10, "FG3A": 3, "FTA": 4})
        players.append({**shared, "PLAYER_ID": 20, "PLAYER_NAME": "Synthetic Player B",
                        "TEAM_ID": 2, "TEAM_ABBREVIATION": "BBB", "PTS": 100 + day,
                        "MIN": 35, "FGA": 15, "FG3A": 5, "FTA": 6})
    births = pd.DataFrame({"PLAYER_ID": [10, 20], "BIRTHDATE": ["2000-01-01", "1995-01-01"]})
    return pd.DataFrame(players), pd.DataFrame(teams), births, pd.DataFrame(snapshots)


def seed_cache(root, season):
    players, teams, births, snapshots = fixture(season)
    dirty = root / "dirty"
    save_csv(players, dirty / season / "players.csv")
    save_csv(teams, dirty / season / "teams.csv")
    for row in births.itertuples():
        save_csv(pd.DataFrame({"PERSON_ID": [row.PLAYER_ID], "BIRTHDATE": [row.BIRTHDATE]}),
                 dirty / "profiles" / f"{row.PLAYER_ID}.csv")
    for cutoff, group in snapshots.groupby("STATS_THROUGH"):
        save_csv(group[["TEAM_ID", "GP", "PACE", "DEF_RATING"]],
                 dirty / season / "team_stats" / f"{cutoff:%Y-%m-%d}.csv")


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.players, self.teams, self.births, self.snapshots = fixture()

    def build(self):
        return build_features(self.players, self.teams, self.births, self.snapshots)

    def test_expected_values_and_player_name(self):
        result = self.build()
        row = result[(result.player_id == 10) & (result.game_date == "2025-01-12")].iloc[0]
        self.assertEqual(row.player_name, "Synthetic José")
        self.assertEqual(row.points_last_5, 9)
        self.assertAlmostEqual(row.points_last_10, 6.1)
        self.assertAlmostEqual(row.points_season_avg, 6.1)
        self.assertEqual(row.games_played_before, 10)
        self.assertEqual(row.minutes_last_10, 30)
        self.assertEqual(row.fga_last_10, 10)
        self.assertEqual(row.fg3a_last_10, 3)
        self.assertEqual(row.fta_last_10, 4)
        self.assertEqual(row.team_pace, 100)
        self.assertEqual(row.opponent_pace, 90)
        self.assertEqual(row.opponent_defensive_rating, 108)
        self.assertEqual(row.opponent_id, 2)
        self.assertEqual(row.target_points, 12)
        self.assertAlmostEqual(row.age, (pd.Timestamp("2025-01-12") - pd.Timestamp("2000-01-01")).days / 365.2425)
        self.assertNotIn("player_name", FEATURE_COLUMNS)
        self.assertNotIn("target_points", FEATURE_COLUMNS)

    def test_rest_is_based_on_team_games_not_player_appearances(self):
        result = self.build()
        row = result[(result.player_id == 10) & (result.game_date == "2025-01-06")].iloc[0]
        self.assertEqual(row.team_rest_days, 0)
        self.assertEqual(row.team_back_to_back, 1)
        self.assertEqual(row.opponent_rest_days, 0)

    def test_current_and_future_outcomes_cannot_change_earlier_inputs(self):
        before = self.build()
        self.players.loc[self.players.GAME_DATE >= "2025-01-11", ["PTS", "MIN", "FGA", "FG3A", "FTA"]] = 999
        after = self.build()
        cutoff = before.game_date <= "2025-01-11"
        pd.testing.assert_frame_equal(before.loc[cutoff, FEATURE_COLUMNS], after.loc[cutoff, FEATURE_COLUMNS])

    def test_trade_preserves_history_but_changes_opponent(self):
        self.players.loc[(self.players.PLAYER_ID == 10) & (self.players.GAME_DATE == "2025-01-12"),
                         ["TEAM_ID", "TEAM_ABBREVIATION"]] = [2, "BBB"]
        result = self.build()
        row = result[(result.player_id == 10) & (result.game_date == "2025-01-12")].iloc[0]
        self.assertAlmostEqual(row.points_last_10, 6.1)
        self.assertEqual(row.opponent_id, 1)
        self.assertEqual(row.is_home, 0)

    def test_history_resets_at_season_boundary(self):
        players, teams, births, snapshots = fixture("2025-26")
        self.players = pd.concat([self.players, players])
        self.teams = pd.concat([self.teams, teams])
        self.snapshots = pd.concat([self.snapshots, snapshots])
        first = self.build().query("games_played_before == 0")
        self.assertEqual(len(first), 4)
        self.assertTrue(first.points_season_avg.isna().all())
        self.assertTrue(first.team_rest_days.isna().all())
        self.assertTrue(first.team_pace.isna().all())

    def test_exclusions_are_explicit_and_remaining_rows_are_complete(self):
        clean, excluded = separate_ready_rows(self.build())
        self.assertEqual(len(clean), 3)
        self.assertEqual(len(excluded), 20)
        self.assertFalse(clean[FEATURE_COLUMNS].isna().any().any())
        self.assertTrue(excluded.missing_columns.str.contains("points_last_10").all())

    def test_duplicate_records_are_rejected(self):
        self.players = pd.concat([self.players, self.players.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate player-game"):
            self.build()

    def test_future_dated_snapshots_are_rejected(self):
        self.snapshots.STATS_THROUGH = self.snapshots.GAME_DATE
        with self.assertRaisesRegex(ValueError, "before the game"):
            self.build()

    def test_ignored_date_filter_is_detected_from_games_played(self):
        self.snapshots["GP"] = 82
        with self.assertRaisesRegex(ValueError, "games-played counts"):
            self.build()

    def test_missing_name_is_rejected(self):
        self.players.loc[0, "PLAYER_NAME"] = " "
        with self.assertRaisesRegex(ValueError, "names must not be empty"):
            self.build()


class PipelineTest(unittest.TestCase):
    def test_offline_csv_export_keeps_names_ids_and_validation_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed_cache(root, "2025-26")
            args = parse_args(["--split", "validation", "--offline", "--data-dir", folder])
            run(args)
            saved = pd.read_csv(root / "clean/validation_2025_2026.csv", dtype={"game_id": "string"})
            self.assertEqual(len(saved), 3)
            self.assertEqual(set(saved.season), {"2025-26"})
            self.assertIn("Synthetic José", set(saved.player_name))
            self.assertTrue(saved.game_id.str.startswith("00").all())
            self.assertFalse((root / "clean/train_2020_2025.csv").exists())
            report = json.loads((root / "clean/report_validation.json").read_text())
            self.assertEqual(report["seasons"][0]["excluded_rows"], 20)

    def test_full_split_membership_and_no_player_game_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for season in ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]:
                seed_cache(root, season)
            run(parse_args(["--split", "all", "--offline", "--data-dir", folder]))
            frames = [pd.read_csv(root / "clean" / filename) for filename in
                      ["train_2020_2025.csv", "validation_2025_2026.csv", "test_2026_2027.csv"]]
            self.assertEqual([len(frame) for frame in frames], [15, 3, 3])
            self.assertFalse(pd.concat(frames).duplicated(["game_id", "player_id"]).any())

    def test_subset_exports_do_not_overwrite_full_study(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed_cache(root, "2025-26")
            run(parse_args(["--split", "validation", "--offline", "--data-dir", folder, "--player-id", "10"]))
            self.assertTrue((root / "clean/samples/players_10/validation_2025_2026.csv").exists())
            self.assertFalse((root / "clean/validation_2025_2026.csv").exists())

    def test_snapshot_requests_are_cached_and_cut_off_before_target(self):
        with tempfile.TemporaryDirectory() as folder:
            source = NBAData(Path(folder))
            with patch("scripts.nba_source.leaguedashteamstats.LeagueDashTeamStats") as endpoint, patch("scripts.nba_source.time.sleep"):
                endpoint.return_value.get_data_frames.return_value = [pd.DataFrame({
                    "TEAM_ID": [1], "GP": [2], "DEF_RATING": [111], "PACE": [99],
                })]
                first = source.team_snapshots("2024-25", ["2025-01-03", "2025-01-03"])
                second = source.team_snapshots("2024-25", ["2025-01-03"])
                self.assertEqual(endpoint.call_count, 1)
                self.assertEqual(endpoint.call_args.kwargs["date_to_nullable"], "01/02/2025")
                pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
