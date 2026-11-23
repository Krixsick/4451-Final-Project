"""Offline checks for chronological evaluation and saved model behavior."""

import argparse
import contextlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.features import FEATURE_COLUMNS
from scripts.train_model import load_dataset, run, train_and_evaluate


def example_rows(start):
    # Invented data for software checks, never for reported model performance.
    count = 60
    frame = pd.DataFrame({column: np.arange(count) % 10 + 1.0
                          for column in FEATURE_COLUMNS})
    frame["season"] = "synthetic"
    frame["game_date"] = pd.date_range(start, periods=count)
    frame["game_id"] = [f"00{start}-{index}" for index in range(count)]
    frame["player_id"] = 1
    frame["player_name"] = "Invented Player"
    frame["target_points"] = np.arange(count) % 20
    return frame


class TrainingTests(unittest.TestCase):
    def test_rejects_overlapping_dates(self):
        with self.assertRaisesRegex(ValueError, "before"):
            train_and_evaluate(example_rows("2020-01-01"), example_rows("2020-02-01"))

    def test_rejects_shared_games_even_with_different_dates(self):
        train = example_rows("2020-01-01")
        validation = example_rows("2021-01-01")
        validation["game_id"] = train["game_id"]
        with self.assertRaisesRegex(ValueError, "overlapping player-games"):
            train_and_evaluate(train, validation)

    def test_rejects_invalid_csv_rows(self):
        original = example_rows("2020-01-01")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            for defect in ["missing", "infinite", "duplicate"]:
                with self.subTest(defect=defect):
                    frame = original.copy()
                    if defect == "missing":
                        frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
                    elif defect == "infinite":
                        frame.loc[0, FEATURE_COLUMNS[0]] = np.inf
                    else:
                        frame = pd.concat([frame, frame.iloc[[0]]])
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_dataset(path)

    def test_export_roundtrip_and_validation_targets_do_not_train_model(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            train = example_rows("2020-01-01")
            validation = example_rows("2021-01-01")
            train.to_csv(folder / "train.csv", index=False)
            validation.to_csv(folder / "validation.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                run(argparse.Namespace(train=folder / "train.csv",
                                       validation=folder / "validation.csv", output=folder / "out"))
            predictions = pd.read_csv(folder / "out/validation_predictions.csv", dtype={"game_id": "string"})
            scores = pd.read_csv(folder / "out/metrics.csv").set_index("method")
            with (folder / "out/model.pkl").open("rb") as source:
                bundle = pickle.load(source)
            self.assertEqual(bundle["feature_columns"], FEATURE_COLUMNS)
            self.assertEqual(list(bundle["model"].feature_names_in_), FEATURE_COLUMNS)
            self.assertEqual(predictions["game_id"].tolist(), validation["game_id"].tolist())
            np.testing.assert_allclose(predictions["season_average"], validation["points_season_avg"])
            expected_mae = (validation["target_points"] - validation["points_season_avg"]).abs().mean()
            self.assertAlmostEqual(scores.loc["season_average", "mae"], expected_mae)
            np.testing.assert_allclose(predictions["model_prediction"], np.maximum(
                bundle["model"].predict(validation[FEATURE_COLUMNS]), 0))
            validation["target_points"] += 100
            validation["player_name"] = "Changed metadata"
            _, changed, _ = train_and_evaluate(train, validation)
            np.testing.assert_allclose(predictions["model_prediction"], changed["model_prediction"])


if __name__ == "__main__":
    unittest.main()
