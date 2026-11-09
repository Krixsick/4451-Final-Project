"""Train a points model and compare it with two averages on later games."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from scripts.features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_points"
METADATA = ["season", "game_id", "game_date", "player_id", "player_name"]


def load_dataset(path):
    """Reject malformed data rather than silently changing the evaluation rows."""
    frame = pd.read_csv(path, dtype={"game_id": "string"})
    required = METADATA + FEATURE_COLUMNS + [TARGET]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f"{path}: dataset is empty or contains missing values")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    if frame["game_date"].isna().any():
        raise ValueError(f"{path}: missing game dates")
    numeric = frame[FEATURE_COLUMNS + [TARGET]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: features and targets must be finite")
    if (numeric[TARGET] < 0).any():
        raise ValueError(f"{path}: target points must be nonnegative")
    frame[FEATURE_COLUMNS + [TARGET]] = numeric
    if frame.duplicated(["game_id", "player_id"]).any():
        raise ValueError(f"{path}: duplicate player-game rows")
    return frame.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def train_and_evaluate(training, validation):
    """Fit only on earlier games; evaluate every method on identical later rows."""
    if training["game_date"].max() >= validation["game_date"].min():
        raise ValueError("All training dates must be before all validation dates")
    keys = ["game_id", "player_id"]
    if not training[keys].merge(validation[keys], on=keys).empty:
        raise ValueError("Training and validation contain overlapping player-games")

    # A single shared model learns from all players. Names and IDs are excluded.
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,  # Avoid an automatic random validation split.
        random_state=42,
    )
    model.fit(training[FEATURE_COLUMNS], training[TARGET])

    predictions = validation[METADATA + [TARGET]].copy()
    predictions["season_average"] = validation["points_season_avg"]
    predictions["last_10_average"] = validation["points_last_10"]
    predictions["model_prediction"] = np.maximum(
        model.predict(validation[FEATURE_COLUMNS]), 0
    )

    scores = []
    for method in ["season_average", "last_10_average", "model_prediction"]:
        scores.append({
            "method": method,
            "mae": mean_absolute_error(validation[TARGET], predictions[method]),
            "rmse": root_mean_squared_error(validation[TARGET], predictions[method]),
        })
    return model, predictions, pd.DataFrame(scores)


def run(args):
    training = load_dataset(args.train)
    validation = load_dataset(args.validation)
    model, predictions, scores = train_and_evaluate(training, validation)

    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "validation_predictions.csv", index=False)
    scores.to_csv(args.output / "metrics.csv", index=False)
    # Keep feature order and postprocessing with the model for later serving.
    bundle = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET,
        "prediction_minimum": 0,
        "sklearn_version": sklearn.__version__,
    }
    with (args.output / "model.pkl").open("wb") as destination:
        pickle.dump(bundle, destination)

    report = {"features": list(FEATURE_COLUMNS), "model_parameters": model.get_params(),
              "sklearn_version": sklearn.__version__, "metrics": scores.to_dict("records")}
    for name, frame, path in [("train", training, args.train),
                              ("validation", validation, args.validation)]:
        report[name] = {
            "file": str(path.resolve()), "rows": len(frame),
            "players": int(frame["player_id"].nunique()),
            "seasons": sorted(frame["season"].unique().tolist()),
            "first_date": str(frame["game_date"].min().date()),
            "last_date": str(frame["game_date"].max().date()),
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved model and results to {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    clean = PROJECT_ROOT / "data" / "clean"
    parser.add_argument("--train", type=Path, default=clean / "train_2020_2025.csv")
    parser.add_argument("--validation", type=Path, default=clean / "validation_2025_2026.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "models" / "model_a")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
