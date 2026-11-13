"""Data and inference boundaries for the official protocol, using invented records."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from blueprint_fixtures import FixtureSource, fixture_tables
from scripts.blueprint import prepare_blueprint
from scripts.experiment_config import SPLITS, SPLIT_FILES
from scripts.features import FEATURE_COLUMNS, build_features
from scripts.model_artifact import prepare_inference_features, read_inference_csv
from scripts.modeling import TARGET, load_clean_split, preparation_report, validate_split_order


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.clean = cls.root / "clean"
        with patch("scripts.blueprint.NBAData", return_value=FixtureSource()), contextlib.redirect_stdout(io.StringIO()):
            prepare_blueprint(cls.root, cls.clean, offline=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def load(self, split):
        return load_clean_split(self.clean / SPLIT_FILES[split], SPLITS[split])

    def changed_path(self, frame):
        path = self.root / "changed.csv"
        frame.to_csv(path, index=False)
        return path

    def test_game_ids_normalize_and_duplicate_keys_fail(self):
        frame = self.load("train")
        frame["game_id"] = frame["game_id"].str.lstrip("0")
        loaded = load_clean_split(self.changed_path(frame), SPLITS["train"])
        self.assertTrue(loaded["game_id"].str.len().eq(10).all())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_clean_split(self.changed_path(pd.concat([frame, frame.iloc[[0]]])), SPLITS["train"])

    def test_invalid_clean_values_and_insufficient_history_fail(self):
        for column, value in [("age", np.inf), ("team_pace", np.nan), (TARGET, -1),
                              ("games_played_before", 9), ("games_played_before", 10.5), ("is_home", 2)]:
            with self.subTest(column=column, value=value):
                frame = self.load("train")
                frame[column] = frame[column].astype(float)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    load_clean_split(self.changed_path(frame), SPLITS["train"])

    def test_raw_schema_and_wrong_seasons_fail(self):
        raw = fixture_tables("2024-25")[0]
        with self.assertRaisesRegex(ValueError, "missing clean columns"):
            load_clean_split(self.changed_path(raw), ["2024-25"])
        with self.assertRaisesRegex(ValueError, "seasons"):
            load_clean_split(self.clean / SPLIT_FILES["train"], SPLITS["validation"])

    def test_game_overlap_and_reversed_chronology_fail(self):
        train, validation = self.load("train"), self.load("validation")
        validate_split_order(train, validation)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_order(train, train)
        with self.assertRaisesRegex(ValueError, "chronological"):
            validate_split_order(validation, train)

    def test_source_clean_excluded_counts_reconcile(self):
        frame = self.load("train")
        _, coverage, _ = preparation_report(self.clean, "train", frame, SPLITS["train"])
        self.assertEqual(int(coverage["source_rows"].sum()), 192)
        self.assertEqual(int(coverage["clean_rows"].sum()), 32)
        self.assertEqual(int(coverage["excluded_rows"].sum()), 160)
        report = json.loads((self.clean / "report_train.json").read_text())
        report["seasons"][0]["clean_rows"] += 1
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "report_train.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "row count"):
                preparation_report(folder, "train", frame, SPLITS["train"])

    def test_current_and_future_outcomes_cannot_change_current_features(self):
        players, teams, births, snapshots = fixture_tables("2024-25")
        original = build_features(players, teams, births, snapshots)
        cutoff = sorted(players["GAME_DATE"].unique())[10]
        changed = players.copy()
        changed.loc[changed["GAME_DATE"].ge(cutoff), ["PTS", "MIN", "FGA", "FG3A", "FTA"]] += 1000
        after = build_features(changed, teams, births, snapshots)
        pd.testing.assert_frame_equal(original.loc[original.game_date.le(cutoff), FEATURE_COLUMNS],
                                      after.loc[after.game_date.le(cutoff), FEATURE_COLUMNS])
        self.assertTrue(after.loc[after.game_date.eq(cutoff), TARGET].eq(
            original.loc[original.game_date.eq(cutoff), TARGET] + 1000).all())

    def test_snapshot_cutoff_and_ignored_date_filter_are_rejected(self):
        players, teams, births, snapshots = fixture_tables("2024-25")
        snapshots["STATS_THROUGH"] = snapshots["GAME_DATE"]
        with self.assertRaisesRegex(ValueError, "before"):
            build_features(players, teams, births, snapshots)
        snapshots = fixture_tables("2024-25")[3]
        snapshots["GP"] = 999
        with self.assertRaisesRegex(ValueError, "games-played"):
            build_features(players, teams, births, snapshots)

    def test_team_rest_uses_team_games_across_player_absences(self):
        players, teams, births, snapshots = fixture_tables("2024-25")
        dates = sorted(players["GAME_DATE"].unique())
        players = players.loc[~(players.PLAYER_ID.eq(101) & players.GAME_DATE.eq(dates[5]))]
        output = build_features(players, teams, births, snapshots)
        returned = output.loc[output.player_id.eq(101) & output.game_date.eq(dates[6])].iloc[0]
        self.assertEqual(returned["team_rest_days"], 1)
        self.assertEqual(returned["games_played_before"], 5)

    def test_inference_rejects_missing_duplicate_and_invalid_features(self):
        frame = self.load("validation").drop(columns=TARGET)
        with self.assertRaisesRegex(ValueError, "Missing"):
            prepare_inference_features(frame.drop(columns="team_pace"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_inference_features(pd.concat([frame, frame[["age"]]], axis=1))
        for column, value in [("age", np.inf), ("team_pace", np.nan), ("fga_last_10", -1),
                              ("is_home", 2), ("games_played_before", 9), ("team_pace", "invalid")]:
            with self.subTest(column=column, value=value):
                changed = frame.copy()
                changed[column] = changed[column].astype(object)
                changed.loc[0, column] = value
                with self.assertRaises(ValueError):
                    prepare_inference_features(changed)

    def test_duplicate_csv_headers_fail_before_pandas_renames_them(self):
        path = self.root / "duplicate_headers.csv"
        path.write_text("age,age\n20,21\n")
        with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
            read_inference_csv(path)


if __name__ == "__main__":
    unittest.main()
