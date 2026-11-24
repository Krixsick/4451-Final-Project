"""Reproduce the locked Ridge fit, verify validation and score the RQ3 holdout."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.experiment_config import FEATURE_GROUPS, FIXED, SPLITS, SPLIT_FILES, protocol_sha256
from scripts.modeling import (METADATA_COLUMNS, TARGET, file_sha256, load_clean_split,
                             preparation_report, regression_metrics, validate_split_order)
from scripts.nba_source import save_csv
from scripts.prepare_data import PROJECT_ROOT
from scripts.prepare_rq3 import CUTOFF, TEST_DIR

TRAIN_SHA = "8dd8238d58def16ef522e55c0b648f4bbc1619830e0dda7c10329639499b6cdb"
VALIDATION_SHA = "667d89f375d6d7c23f8f3deb0acbaa2c6711aec51273199e38c132eed54e99b5"
VALIDATION_RIDGE = (4.6573056804417625, 6.049599141422061)
FEATURES = FEATURE_GROUPS["F1"]
METHODS = {"Ridge F1": "ridge_points", "Ten-game mean": "baseline_points"}


def scores(frame):
    return [{"method": label, **regression_metrics(frame[TARGET], frame[column])}
            for label, column in METHODS.items()]


def predict(model, frame):
    result = frame[METADATA_COLUMNS + [TARGET]].copy()
    result["ridge_points"] = model.predict(frame[FEATURES])
    result["baseline_points"] = frame["points_last_10"]
    return result


def evaluate(train_path, validation_path, test_dir, output):
    if output.exists():
        raise FileExistsError(f"Result directory already exists: {output}")
    # Appendix C.4 identifies the exact team inputs for the training-only refit.
    for path, expected in [(train_path, TRAIN_SHA), (validation_path, VALIDATION_SHA)]:
        if file_sha256(path) != expected:
            raise ValueError(f"{path} differs from the recorded Blueprint data")
    train = load_clean_split(train_path, SPLITS["train"])
    validation = load_clean_split(validation_path, SPLITS["validation"])
    validate_split_order(train, validation)
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=100.0, **FIXED["M1"]))])
    with threadpool_limits(limits=2):
        model.fit(train[FEATURES], train[TARGET])
        training_scores = scores(predict(model, train))
        validation_scores = scores(predict(model, validation))
    ridge = validation_scores[0]
    np.testing.assert_allclose([ridge["MAE"], ridge["RMSE"]], VALIDATION_RIDGE, atol=1e-10, rtol=0)

    test_path = test_dir / SPLIT_FILES["test"]
    test = load_clean_split(test_path, SPLITS["test"])
    validate_split_order(validation, test)
    audit, coverage, report_path = preparation_report(test_dir, "test", test, SPLITS["test"])
    if (not audit or audit.get("cutoff") != CUTOFF
            or pd.Timestamp(coverage.iloc[0]["last_source_game_date"]) > pd.Timestamp(CUTOFF)):
        raise ValueError("Test preparation must cover the registered window")
    with threadpool_limits(limits=2):
        predictions = predict(model, test)
    metrics = pd.DataFrame([{"split": split, **row}
                            for split, rows in [("train", training_scores), ("validation", validation_scores),
                                                ("test", scores(predictions))]
                            for row in rows])

    distributions = []
    for label, column in METHODS.items():
        error = predictions[column] - predictions[TARGET]
        predictions[column + "_error"] = error
        for measure, series in [("signed", error), ("absolute", error.abs())]:
            quantiles = series.quantile([0, .1, .25, .5, .75, .9, .95, 1])
            distributions.append({"method": label, "error": measure, "mean": series.mean(),
                                  **{f"p{round(100*q)}": value for q, value in quantiles.items()}})

    grouping = {
        "month": test["game_date"].dt.strftime("%Y-%m"),
        "prior_10_minutes": pd.cut(test["minutes_last_10"], [0, 20, 30, np.inf],
                                   right=False, labels=["under 20", "20 to under 30", "30 or more"]),
        "location": test["is_home"].map({0: "away", 1: "home"}),
        "team_back_to_back": test["team_back_to_back"].map({0: "no", 1: "yes"}),
    }
    groups = [{"group": name, "value": value, **row}
              for name, labels in grouping.items()
              for value, group in predictions.groupby(labels, observed=True)
              for row in scores(group)]
    summary = {
        "candidate": "M1_F1", "alpha": 100.0, "solver": "svd", "feature_columns": FEATURES,
        "protocol_sha256": protocol_sha256(), "train_sha256": TRAIN_SHA, "validation_sha256": VALIDATION_SHA,
        "training_rows": len(train), "validation_rows": len(validation),
        "fit_policy": "Exact training CSV; fixed training-only refit; validation metrics reproduced",
        "test_sha256": file_sha256(test_path), "test_report_sha256": file_sha256(report_path),
        "cutoff": CUTOFF, "source_first_date": audit["first_source_game_date"],
        "source_games": audit["source_games"], "coverage": coverage.iloc[0].to_dict(),
        "first_eligible_date": str(test["game_date"].min().date()),
        "last_eligible_date": str(test["game_date"].max().date()),
        "eligible_games": int(test["game_id"].nunique()), "eligible_players": int(test["player_id"].nunique()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "result"
        for filename, frame in [("metrics", metrics), ("predictions", predictions),
                                ("error_distribution", pd.DataFrame(distributions)),
                                ("stratified_metrics", pd.DataFrame(groups))]:
            save_csv(frame, stage / f"{filename}.csv")
        (stage / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(output)
    return metrics


if __name__ == "__main__":
    clean = PROJECT_ROOT / "data/clean/blueprint_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=clean / SPLIT_FILES["train"])
    parser.add_argument("--validation-csv", type=Path, default=clean / SPLIT_FILES["validation"])
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "artifacts/blueprint_v1" / f"rq3_through_{CUTOFF}")
    args = parser.parse_args()
    print(evaluate(args.train_csv, args.validation_csv, args.test_dir, args.output_dir).to_string(index=False))
