"""Freeze and reload any Blueprint winner without retuning or opening test data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from scripts.blueprint import build_estimator
from scripts.experiment_config import CANDIDATES, FEATURE_GROUPS, FIXED, SPLITS, SPLIT_FILES, protocol_sha256
from scripts.model_artifact import runtime_versions
from scripts.modeling import TARGET, file_sha256, load_clean_split, preparation_report, regression_metrics, validate_split_order
from scripts.prepare_data import PROJECT_ROOT

BASELINES = {"M0_previous_10": "points_last_10", "M0_season": "points_season_avg"}


class HistoricalAverage:
    """A non-ML rule; the source column already excludes the target game."""
    def __init__(self, column):
        self.column = column

    def predict(self, features):
        return features[self.column].to_numpy(dtype=float)


def validate_contract(manifest):
    if manifest.get("protocol_sha256") != protocol_sha256() or manifest.get("target_column") != TARGET:
        raise ValueError("Artifact protocol/target differs from this repository")
    model_id, group, variant = (manifest.get(key) for key in ["model_id", "feature_set", "candidate"])
    if model_id == "M0":
        if variant not in BASELINES or group != "baseline":
            raise ValueError("Unknown baseline contract")
        expected = [BASELINES[variant], "games_played_before"]
        filename = "baseline.json"
    else:
        if model_id not in FIXED or group not in FEATURE_GROUPS or variant != f"{model_id}_{group}":
            raise ValueError("Unknown ML feature contract")
        expected = FEATURE_GROUPS[group]
        filename = "model.json" if model_id == "M3" else "model.joblib"
        params = manifest.get("selected_parameters", {})
        if params not in [{**FIXED[model_id], **row} for row in CANDIDATES[model_id]]:
            raise ValueError("Artifact parameters are outside the declared protocol")
    if manifest.get("feature_columns") != expected or manifest.get("model_filename") != filename:
        raise ValueError("Artifact feature order or model filename differs from the contract")


def load_blueprint_artifact(directory, manifest=None):
    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory / "model_manifest.json").read_text())
    if manifest.get("artifact_schema_version") != 2:
        raise ValueError("Unsupported Blueprint artifact schema")
    validate_contract(manifest)
    current = runtime_versions()
    required = ["python_major_minor", "numpy", "pandas", "scikit_learn", "joblib"]
    if manifest["model_id"] == "M3":
        required.append("xgboost")
    for name in required:
        if manifest["runtime_versions"].get(name) != current[name]:
            raise ValueError(f"Runtime mismatch for {name}; use the locked environment")
    path = directory / manifest["model_filename"]
    if file_sha256(path) != manifest["model_sha256"]:
        raise ValueError("Model hash differs from the manifest")
    model_id = manifest["model_id"]
    if model_id == "M0":
        source = json.loads(path.read_text())["source_feature"]
        if source != BASELINES[manifest["candidate"]]:
            raise ValueError("Baseline rule differs from the manifest")
        model = HistoricalAverage(source)
    elif model_id == "M3":
        model = XGBRegressor(**manifest["selected_parameters"])
        model.load_model(path)
    else:
        model = joblib.load(path)  # Only trusted team artifacts, after the hash check.
        expected_type = Pipeline if model_id == "M1" else RandomForestRegressor
        if not isinstance(model, expected_type):
            raise ValueError("Unexpected estimator type")
        actual = model.named_steps["ridge"].get_params() if model_id == "M1" else model.get_params()
        if any(actual.get(k) != v for k, v in manifest["selected_parameters"].items()):
            raise ValueError("Estimator parameters differ from the manifest")
    if model_id != "M0" and list(model.feature_names_in_) != manifest["feature_columns"]:
        raise ValueError("Estimator feature order differs from the manifest")
    return model, manifest


def export_blueprint(clean_dir, results_dir, output_dir):
    clean_dir, results_dir, output_dir = map(Path, [clean_dir, results_dir, output_dir])
    if output_dir.exists():
        raise FileExistsError("Choose a new frozen artifact directory")
    source_path = results_dir / "experiment_manifest.json"
    source = json.loads(source_path.read_text())
    if source.get("protocol_sha256") != protocol_sha256() or not source.get("is_full_protocol_grid"):
        raise ValueError("Export requires the complete official protocol")
    if source.get("final_test_loaded") or source.get("final_test_evaluated"):
        raise ValueError("Source experiment has already opened the final test")
    # Recompute the declared selection rule, rather than trusting a winner label alone.
    scores = pd.DataFrame(source["validation_scores"])
    selected = scores.sort_values(["MAE", "feature_count", "model_id", "feature_set", "variant"]).iloc[0]
    if selected["variant"] != source["selected_candidate"]["variant"]:
        raise ValueError("Candidate disagrees with the declared selection rule")
    variant, model_id, group = selected["variant"], selected["model_id"], selected["feature_set"]
    params = {} if model_id == "M0" else source["selected_parameters"][variant]
    features = [BASELINES[variant], "games_played_before"] if model_id == "M0" else FEATURE_GROUPS[group]
    contract = {"artifact_schema_version": 2, "protocol_sha256": protocol_sha256(),
        "model_id": model_id, "feature_set": group, "candidate": variant,
        "feature_columns": features, "target_column": TARGET, "selected_parameters": params,
        "model_filename": "baseline.json" if model_id == "M0" else "model.json" if model_id == "M3" else "model.joblib"}
    validate_contract(contract)
    frames = {}
    for split in ["train", "validation"]:
        path = clean_dir / SPLIT_FILES[split]
        expected = next(row for row in source["inputs"] if row["split"] == split)
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"{split} data hash differs from the source experiment")
        frames[split] = load_clean_split(path, SPLITS[split])
        report, _, report_path = preparation_report(clean_dir, split, frames[split], SPLITS[split])
        if report is None or file_sha256(report_path) != expected["report_sha256"]:
            raise ValueError("Preparation report differs from the source experiment")
    train, validation = frames["train"], frames["validation"]
    validate_split_order(train, validation)
    prediction_path = results_dir / "validation_predictions.csv"
    frozen = pd.read_csv(prediction_path, dtype={"game_id": "string"})
    frozen["game_date"] = pd.to_datetime(frozen["game_date"]).dt.normalize()
    keys = ["game_id", "player_id"]
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    validation = validation.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(frozen[keys + ["game_date", TARGET]],
        validation[keys + ["game_date", TARGET]], check_dtype=False)
    model = HistoricalAverage(BASELINES[variant]) if model_id == "M0" else build_estimator(
        model_id, {key: params[key] for key in CANDIDATES[model_id][0]})
    with threadpool_limits(limits=2):
        if model_id != "M0":
            model.fit(train[features], train[TARGET])
        predicted = model.predict(validation[features])
    np.testing.assert_allclose(predicted, frozen[variant], rtol=1e-10, atol=1e-10)
    metrics = regression_metrics(validation[TARGET], predicted)
    np.testing.assert_allclose([metrics["MAE"], metrics["RMSE"]],
        [selected["MAE"], selected["RMSE"]], rtol=1e-10)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".blueprint-export-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        path = stage / contract["model_filename"]
        if model_id == "M0":
            path.write_text(json.dumps({"source_feature": BASELINES[variant]}))
        elif model_id == "M3":
            model.save_model(path)
        else:
            joblib.dump(model, path, compress=3)
        manifest = {**contract, "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_sha256": file_sha256(path), "runtime_versions": runtime_versions(),
            "training_seasons": SPLITS["train"], "training_rows": len(train),
            "training_last_game_date": str(train["game_date"].max().date()),
            "validation_seasons": SPLITS["validation"], "validation_rows": len(validation),
            "validation_last_game_date": str(validation["game_date"].max().date()),
            "source_inputs": source["inputs"], "source_experiment_manifest_sha256": file_sha256(source_path),
            "source_validation_predictions_sha256": file_sha256(prediction_path),
            "code_sha256": {name: file_sha256(PROJECT_ROOT / "scripts" / name) for name in
                ["blueprint_artifact.py", "model_artifact.py", "predict.py", "experiment_config.py"]},
            "uv_lock_sha256": file_sha256(PROJECT_ROOT / "uv.lock"),
            "postprocessing": "No clipping or rounding", "participation_scope": "Conditional on player appearance",
            "input_semantics": "Pregame features only; at least ten prior same-season appearances; complete cases",
            "validation_metrics": metrics, "final_test_loaded": False, "final_test_evaluated": False}
        (stage / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        from scripts.model_artifact import load_artifact, predict_frame
        reloaded, manifest = load_artifact(stage)
        after = predict_frame(validation.drop(columns=TARGET), reloaded, manifest)["predicted_points"].to_numpy()
        # Parallel forest reductions can change the last floating-point bit.
        np.testing.assert_allclose(predicted, after, rtol=1e-12, atol=1e-12)
        verification = {"validation_rows": len(validation), "model_fit_rows": len(train) if model_id != "M0" else 0,
            "max_absolute_difference_vs_frozen_predictions": float(np.max(np.abs(predicted - frozen[variant]))),
            "max_absolute_difference_after_reload": float(np.max(np.abs(predicted - after))),
            "validation_metrics": metrics, "refit_on_validation": False, "hyperparameters_retuned": False,
            "final_test_loaded": False, "final_test_evaluated": False}
        (stage / "verification.json").write_text(json.dumps(verification, indent=2))
        if output_dir.exists():
            raise FileExistsError("Output directory appeared during export")
        stage.rename(output_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=PROJECT_ROOT / "data/clean/blueprint_v1")
    parser.add_argument("--source-results-dir", type=Path, default=PROJECT_ROOT / "artifacts/blueprint_v1/results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/blueprint_v1/frozen_model")
    args = parser.parse_args(argv)
    manifest = export_blueprint(args.clean_dir, args.source_results_dir, args.output_dir)
    print(f"Frozen {manifest['candidate']}: {manifest['validation_metrics']}")


if __name__ == "__main__":
    main()
