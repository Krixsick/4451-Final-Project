"""Invented data verify the official protocol and every winner's artifact contract."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from scripts.blueprint import build_estimator, prepare_blueprint, run_blueprint
from scripts.blueprint_artifact import export_blueprint, BASELINES
from scripts.blueprint_report import read_verified_results
from scripts.experiment_config import CANDIDATES, FEATURE_GROUPS, FIXED, SPLITS, SPLIT_FILES, protocol_sha256
from scripts.features import FEATURE_COLUMNS
from scripts.model_artifact import load_artifact, predict_frame, runtime_versions
from scripts.modeling import TARGET, file_sha256, load_clean_split
from scripts.predict import predict_csv
from blueprint_fixtures import FixtureSource, execute_notebook


class BlueprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.clean, cls.results = cls.root / "clean", cls.root / "results"
        with patch("scripts.blueprint.NBAData", return_value=FixtureSource()), contextlib.redirect_stdout(io.StringIO()):
            prepare_blueprint(cls.root, cls.clean, offline=True)
        (cls.clean / SPLIT_FILES["test"]).write_text("TEST MUST NEVER BE READ")
        (cls.clean / "report_test.json").write_text("INVALID TEST REPORT")
        cls.train = load_clean_split(cls.clean / SPLIT_FILES["train"], SPLITS["train"])
        cls.validation = load_clean_split(cls.clean / SPLIT_FILES["validation"], SPLITS["validation"])
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = run_blueprint(cls.clean, cls.results)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_splits_and_nested_pregame_groups(self):
        self.assertEqual(SPLITS["train"], ["2021-22", "2022-23", "2023-24", "2024-25"])
        self.assertEqual(len(self.train), 32)
        self.assertEqual(len(self.validation), 8)
        self.assertEqual([len(group) for group in FEATURE_GROUPS.values()], [10, 12, 18])
        self.assertLess(set(FEATURE_GROUPS["F1"]), set(FEATURE_GROUPS["F2"]))
        self.assertLess(set(FEATURE_GROUPS["F2"]), set(FEATURE_GROUPS["F3"]))
        self.assertEqual(FEATURE_GROUPS["F3"], FEATURE_COLUMNS)
        self.assertTrue(all(TARGET not in cols and "player_id" not in cols for cols in FEATURE_GROUPS.values()))
        self.assertFalse((self.clean / "train_2020_2025.csv").exists())
        with self.assertRaises(FileExistsError):
            prepare_blueprint(self.root, self.clean, offline=True)

    def test_four_model_rq1_and_nine_matched_ablation_variants(self):
        result = self.result
        self.assertEqual(result["rq1"]["model_id"].tolist(), ["M0", "M1", "M2", "M3"])
        self.assertEqual(len(result["rq2"]), 9)
        self.assertEqual(len(result["tuning"]), 39)
        self.assertTrue(result["scores"]["n"].eq(8).all())
        self.assertEqual(len(result["predictions"]), 8)
        self.assertIsInstance(result["models"]["M2_F1"], RandomForestRegressor)
        self.assertIsInstance(result["models"]["M3_F3"], XGBRegressor)
        for group, features in FEATURE_GROUPS.items():
            scaler = result["models"][f"M1_{group}"].named_steps["scale"]
            np.testing.assert_allclose(scaler.mean_, self.train[features].mean())
            self.assertEqual(scaler.n_samples_seen_, len(self.train))
        for _, rows in result["rq2"].groupby("model_id"):
            np.testing.assert_allclose(rows["incremental_MAE_reduction"].iloc[1:], -np.diff(rows["MAE"]))
        self.assertTrue(result["manifest"]["is_full_protocol_grid"])
        self.assertFalse(result["manifest"]["final_test_loaded"])

    def test_saved_results_verify_inputs_and_metrics_without_test(self):
        tables = read_verified_results(self.clean, self.results)
        self.assertEqual(len(tables["rq2_feature_ablation"]), 9)
        with self.assertRaises(FileExistsError):
            run_blueprint(self.clean, self.results)

    def test_export_selected_candidate_train_only_and_target_free(self):
        output = self.root / "export"
        manifest = export_blueprint(self.clean, self.results, output)
        self.assertEqual(manifest["training_rows"], 32)
        self.assertEqual(manifest["training_seasons"], SPLITS["train"])
        self.assertFalse(manifest["final_test_loaded"])
        model, manifest = load_artifact(output)
        frame = self.validation.drop(columns=TARGET)
        expected = predict_frame(frame, model, manifest)
        shuffled = frame[list(reversed(frame.columns))].assign(target_points="IGNORE OUTCOME", PTS=-999)
        np.testing.assert_array_equal(predict_frame(shuffled, model, manifest)["predicted_points"], expected["predicted_points"])
        with self.assertRaises(FileExistsError):
            export_blueprint(self.clean, self.results, output)
        verification = json.loads((output / "verification.json").read_text())
        self.assertEqual(verification["max_absolute_difference_after_reload"], 0)
        self.assertFalse(verification["refit_on_validation"])
        self.assertFalse(verification["hyperparameters_retuned"])

    def test_all_four_families_reload_with_subset_feature_contracts(self):
        # These are explicitly synthetic serializer fixtures, not research selections.
        for model_id in ["M0", "M1", "M2", "M3"]:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                group = "baseline" if model_id == "M0" else "F2"
                variant = "M0_previous_10" if model_id == "M0" else f"{model_id}_{group}"
                features = [BASELINES[variant], "games_played_before"] if model_id == "M0" else FEATURE_GROUPS[group]
                params = {} if model_id == "M0" else {**FIXED[model_id], **CANDIDATES[model_id][0]}
                filename = "baseline.json" if model_id == "M0" else "model.json" if model_id == "M3" else "model.joblib"
                path = directory / filename
                if model_id == "M0":
                    path.write_text(json.dumps({"source_feature": BASELINES[variant]}))
                    expected = self.validation[BASELINES[variant]].to_numpy()
                else:
                    model = build_estimator(model_id, CANDIDATES[model_id][0])
                    model.fit(self.train[features], self.train[TARGET])
                    expected = model.predict(self.validation[features])
                    if model_id == "M3":
                        model.save_model(path)
                    else:
                        joblib.dump(model, path)
                manifest = {"artifact_schema_version": 2, "protocol_sha256": protocol_sha256(), "model_id": model_id,
                    "feature_set": group, "candidate": variant, "feature_columns": features, "target_column": TARGET,
                    "selected_parameters": params, "model_filename": filename, "model_sha256": file_sha256(path),
                    "runtime_versions": runtime_versions()}
                (directory / "model_manifest.json").write_text(json.dumps(manifest))
                loaded, manifest = load_artifact(directory)
                result = predict_frame(self.validation[list(reversed(features))], loaded, manifest)
                np.testing.assert_allclose(result["predicted_points"], expected, rtol=1e-12, atol=1e-12)
                input_path = directory / "unlabeled.csv"
                self.validation[features].to_csv(input_path, index=False)
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    predict_csv(directory, input_path, path)
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_artifact(directory)

    def test_official_notebook_executes_cached_evidence_without_test(self):
        import nbformat
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/NBA_Blueprint_Experiments.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook.cells))
        for cell in notebook.cells:
            if "configuration" in cell.metadata.get("tags", []):
                cell.source = cell.source.replace('CLEAN_DIR = REPO_ROOT / "data" / "clean" / "blueprint_v1"',
                    f'CLEAN_DIR = Path({str(self.clean)!r})')
                cell.source = cell.source.replace('RESULTS_DIR = REPO_ROOT / "artifacts" / "blueprint_v1" / "results"',
                    f'RESULTS_DIR = Path({str(self.results)!r})')
                cell.source = cell.source.replace('MODEL_DIR = REPO_ROOT / "artifacts" / "blueprint_v1" / "frozen_model"',
                    f'MODEL_DIR = Path({str(self.root / "isolated_notebook_model")!r})')
        execute_notebook(notebook, notebook_path.parent)
        self.assertTrue((self.results / "figures/rq1_actual_vs_predicted.png").is_file())


if __name__ == "__main__":
    unittest.main()
