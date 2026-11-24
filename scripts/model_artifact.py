"""Load the team's frozen estimator and validate target-free pregame inputs."""

import csv
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.features import FEATURE_COLUMNS
from scripts.modeling import METADATA_COLUMNS, TARGET, file_sha256

SCHEMA_VERSION = 1


def runtime_versions():
    return {"python": platform.python_version(), "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__, "xgboost": xgboost.__version__}


def load_artifact(model_dir):
    """Load a trusted team artifact after contract, runtime, and file-hash checks."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") == 2:
        from scripts.blueprint_artifact import load_blueprint_artifact
        return load_blueprint_artifact(model_dir, manifest)
    if manifest.get("artifact_schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported model artifact schema")
    if manifest.get("feature_columns") != FEATURE_COLUMNS or manifest.get("target_column") != TARGET:
        raise ValueError("Model feature/target contract differs from this repository version")
    if manifest.get("model_filename") != "model.joblib":
        raise ValueError("Unexpected model filename")
    current = runtime_versions()
    for name in ["python_major_minor", "numpy", "pandas", "scikit_learn", "joblib"]:
        if manifest["runtime_versions"].get(name) != current[name]:
            raise ValueError(f"Runtime mismatch for {name}; use the artifact's locked environment")
    path = model_dir / "model.joblib"
    if file_sha256(path) != manifest.get("model_sha256"):
        raise ValueError("Model file hash does not match model_manifest.json")
    model = joblib.load(path)
    if not isinstance(model, HistGradientBoostingRegressor):
        raise ValueError("Unexpected estimator type")
    if list(model.feature_names_in_) != FEATURE_COLUMNS or model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("Estimator input contract differs from the manifest")
    if model.get_params(deep=False) != manifest["estimator_parameters"]:
        raise ValueError("Estimator parameters differ from the manifest")
    if not hasattr(model, "n_iter_") or model.n_iter_ != manifest["selected_parameters"]["max_iter"]:
        raise ValueError("Estimator is incomplete or has an unexpected iteration count")
    return model, manifest


def prepare_inference_features(frame, features=FEATURE_COLUMNS):
    """Reorder by canonical names; no target, imputation, or scaling is required."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required pregame features: {missing}")
    if frame.empty:
        raise ValueError("Prediction input has no rows")
    values = frame.loc[:, features].copy()
    for column in features:
        try:
            values[column] = pd.to_numeric(values[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Feature {column} must contain numeric values") from error
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Pregame features must be finite, nonnegative, and complete")
    for column in ["is_home", "team_back_to_back", "opponent_back_to_back"]:
        if column in values and not values[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be 0 or 1")
    history = values["games_played_before"]
    if history.lt(10).any() or history.mod(1).ne(0).any():
        raise ValueError("At least ten prior same-season appearances are required")
    return values


def predict_frame(frame, model, manifest):
    """Preserve input row order/optional metadata and ignore all outcome columns."""
    if manifest.get("artifact_schema_version") == 2:
        from scripts.blueprint_artifact import validate_contract
        validate_contract(manifest)
    elif manifest.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Prediction feature order differs from the artifact contract")
    frame = frame.reset_index(drop=True).copy()
    features = prepare_inference_features(frame, manifest["feature_columns"])
    result = frame[[column for column in METADATA_COLUMNS if column in frame]].copy()
    if "game_id" in result:
        ids = result["game_id"].astype("string")
        if not ids.str.fullmatch(r"[0-9]{1,10}").fillna(False).all():
            raise ValueError("game_id metadata must contain 1–10 digits")
        result["game_id"] = ids.str.zfill(10)
    if {"game_id", "player_id"}.issubset(result):
        if result[["game_id", "player_id"]].isna().any().any() or result.duplicated(["game_id", "player_id"]).any():
            raise ValueError("Duplicate or missing player-game metadata keys")
    result.insert(0, "input_row", np.arange(len(frame)))
    with threadpool_limits(limits=2):
        predicted = model.predict(features)
    if not np.isfinite(predicted).all():
        raise ValueError("Estimator returned non-finite predictions")
    result["predicted_points"] = predicted
    return result


def read_inference_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header:
        raise ValueError("Input CSV has no header")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    return pd.read_csv(path, dtype={"game_id": "string"})
