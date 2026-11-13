"""Authoritative team Blueprint v1 protocol; earlier experiment settings are legacy."""

import hashlib
import itertools
import json

from scripts.features import FEATURE_COLUMNS

PROTOCOL_ID = "blueprint_v1"
SPLITS = {
    "train": ["2021-22", "2022-23", "2023-24", "2024-25"],
    "validation": ["2025-26"], "test": ["2026-27"],
}
SPLIT_FILES = {"train": "train_2021_2025.csv", "validation": "validation_2025_2026.csv", "test": "test_2026_2027.csv"}
F1 = ["points_last_5", "points_last_10", "points_season_avg", "minutes_last_5", "minutes_last_10",
      "fga_last_10", "fg3a_last_10", "fta_last_10", "age", "games_played_before"]
FEATURE_GROUPS = {"F1": F1, "F2": F1 + ["opponent_defensive_rating", "opponent_pace"],
                  "F3": list(FEATURE_COLUMNS)}
MODEL_LABELS = {"M0": "Historical average", "M1": "Scaled Ridge regression",
                "M2": "Random Forest", "M3": "XGBoost (CPU hist)"}
FIXED = {
    "M1": {"solver": "svd"},
    "M2": {"n_estimators": 150, "criterion": "squared_error", "max_features": 1.0,
           "bootstrap": True, "random_state": 42, "n_jobs": 2},
    "M3": {"n_estimators": 300, "learning_rate": 0.05, "min_child_weight": 10,
           "subsample": 1.0, "colsample_bytree": 1.0, "objective": "reg:squarederror",
           "tree_method": "hist", "random_state": 42, "n_jobs": 2, "verbosity": 0},
}
CANDIDATES = {
    "M1": [{"alpha": value} for value in [0.1, 1.0, 10.0, 100.0, 1000.0]],
    "M2": [{"max_depth": depth, "min_samples_leaf": leaf}
           for depth, leaf in itertools.product([8, 12], [10, 30])],
    "M3": [{"max_depth": depth, "reg_lambda": penalty}
           for depth, penalty in itertools.product([3, 5], [1.0, 10.0])],
}


def protocol_spec():
    return {"protocol_id": PROTOCOL_ID, "splits": SPLITS, "split_files": SPLIT_FILES,
        "feature_groups": FEATURE_GROUPS, "models": MODEL_LABELS, "fixed_parameters": FIXED,
        "candidate_grids": CANDIDATES, "primary_metric": "MAE", "secondary_metric": "RMSE",
        "cohort": "Identical upstream 18-input complete cases; at least ten prior same-season appearances",
        "RQ1": "Validation-selected M0 baseline and M1/M2/M3 with F3",
        "RQ2": "Independently validation-tuned M1/M2/M3 at F1, F2, F3; matched cohorts",
        "selection": "Lowest validation MAE across M0 variants and all nine ML variants; ties favor fewer inputs then model/group order",
        "candidate_tie": "Original candidate grid order", "fit_policy": "Training only; no validation refit or early stopping",
        "RQ3": "Pending; test outcomes held out until candidate lock"}


def protocol_sha256():
    return hashlib.sha256(json.dumps(protocol_spec(), sort_keys=True).encode()).hexdigest()
