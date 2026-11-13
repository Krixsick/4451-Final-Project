"""Clean-split contracts and scoring helpers; histories are prepared upstream."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.features import FEATURE_COLUMNS


METADATA_COLUMNS = [
    "season", "game_id", "game_date", "player_id", "player_name",
    "team_id", "team_abbreviation", "opponent_id",
]
TARGET = "target_points"
METHODS = ["previous_10_game_average", "season_to_date_average", "ridge"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_split(path, expected_seasons):
    """Require complete prepare_data output; never silently repair model inputs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Clean dataset not found: {path.resolve()}. Run "
            "'uv run --frozen python -m scripts.blueprint prepare' for the official "
            "Blueprint splits, or point CLEAN_DIR at the matching clean exports. "
            "Traditional player_game_logs CSVs are not interchangeable with clean features."
        )
    data = pd.read_csv(path, dtype={"game_id": "string", "season": "string"})
    required = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{path.name}: missing clean columns {missing}; regenerate with prepare_data")
    if data.empty or data[required].isna().any().any():
        raise ValueError(f"{path.name}: empty or incomplete clean data; inspect preparation exclusions")
    if not data["game_id"].str.fullmatch(r"[0-9]{1,10}").all():
        raise ValueError("game_id must contain 1–10 digits, not floating-point IDs")
    data["game_id"] = data["game_id"].str.zfill(10)
    data["game_date"] = pd.to_datetime(data["game_date"], errors="raise").dt.normalize()
    if set(data["season"].unique()) != set(expected_seasons):
        raise ValueError(f"{path.name}: seasons disagree with the configured protocol")
    if data.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Duplicate player-game keys after ID normalization")
    if data.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("A game_id has inconsistent dates")
    if data["player_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing player display name")
    for column in FEATURE_COLUMNS + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    numeric = data[FEATURE_COLUMNS + [TARGET]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Clean features and target must be finite and nonnegative")
    prior = data["games_played_before"]
    if prior.lt(10).any() or prior.mod(1).ne(0).any():
        raise ValueError("Clean rows require at least ten prior same-season appearances")
    for column in ["is_home", "team_back_to_back", "opponent_back_to_back"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be 0 or 1")
    return data.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)


def validate_split_order(earlier, later):
    """Keep entire games and seasons disjoint and ordered in time."""
    if earlier.empty or later.empty:
        raise ValueError("Cannot evaluate an empty split")
    if set(earlier["game_id"]) & set(later["game_id"]):
        raise ValueError("Game overlap between splits")
    if set(earlier["season"]) & set(later["season"]):
        raise ValueError("Season overlap between splits")
    if earlier["game_date"].max() >= later["game_date"].min():
        raise ValueError("Splits must be strictly chronological")


def preparation_report(clean_dir, split, data, split_seasons):
    """Read split-specific or report_all audit; do not guess exclusion counts."""
    clean_dir = Path(clean_dir)
    specific = clean_dir / f"report_{split}.json"
    path = specific if specific.exists() else clean_dir / "report_all.json"
    if not path.exists():
        return None, pd.DataFrame(), None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("feature_columns") != FEATURE_COLUMNS or report.get("target_column") != TARGET:
        raise ValueError(f"{path.name}: report feature/target contract differs from the repository")
    if report.get("player_ids"):
        raise ValueError("Player-sample report supplied for a league-wide modeling experiment")
    if report.get("split_seasons", {}).get(split) != list(split_seasons):
        raise ValueError(f"{path.name}: report split differs from the configured protocol")
    seasons = report.get("seasons", [])
    selected = [entry for entry in seasons if entry.get("season") in split_seasons]
    if len(selected) != len(split_seasons) or {r["season"] for r in selected} != set(split_seasons):
        raise ValueError(f"{path.name}: missing or duplicate season audits")
    rows = []
    for entry in selected:
        season_data = data.loc[data["season"].eq(entry["season"])]
        if len(season_data) != entry["clean_rows"]:
            raise ValueError(f"{path.name}: clean row count disagrees with CSV")
        if entry["source_rows"] != entry["clean_rows"] + entry["excluded_rows"]:
            raise ValueError(f"{path.name}: source/clean/excluded counts do not reconcile")
        if not season_data.empty and season_data["game_date"].max() > pd.Timestamp(entry["last_game_date"]):
            raise ValueError(f"{path.name}: CSV extends beyond reported last game")
        source_rows = entry["source_rows"]
        rows.append({"split": split, "season": entry["season"],
                     "source_rows": source_rows, "clean_rows": entry["clean_rows"],
                     "excluded_rows": entry["excluded_rows"],
                     "coverage_pct": 100 * entry["clean_rows"] / source_rows if source_rows else None,
                     "last_source_game_date": entry["last_game_date"]})
    return report, pd.DataFrame(rows), path


def regression_metrics(actual, predicted):
    return {"n": len(actual), "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted)))}


def fit_ridge(train, validation, alphas, feature_columns=None):
    """Fit only train; select alpha by validation MAE, smaller alpha breaks ties."""
    validate_split_order(train, validation)
    features = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    if not features or len(features) != len(set(features)) or not set(features).issubset(FEATURE_COLUMNS):
        raise ValueError("Ridge inputs must be a nonempty unique subset of FEATURE_COLUMNS")
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Provide a nonempty list of positive finite Ridge alphas")
    models, rows = {}, []
    for alpha in sorted(set(alphas)):
        model = Pipeline([("scale", StandardScaler()),
                          ("ridge", Ridge(alpha=alpha, solver="svd"))])
        model.fit(train[features], train[TARGET])
        predicted = model.predict(validation[features])
        rows.append({"alpha": alpha, **regression_metrics(validation[TARGET], predicted)})
        models[alpha] = model
    tuning = pd.DataFrame(rows).sort_values(["MAE", "alpha"]).reset_index(drop=True)
    alpha = float(tuning.iloc[0]["alpha"])
    return models[alpha], alpha, tuning


def prediction_table(data, ridge_model):
    result = data[METADATA_COLUMNS + ["games_played_before", TARGET]].copy()
    result["previous_10_game_average"] = data["points_last_10"]
    result["season_to_date_average"] = data["points_season_avg"]
    result["ridge"] = ridge_model.predict(data[FEATURE_COLUMNS])
    return result


def score_predictions(predictions):
    metrics = pd.DataFrame([
        {"method": method, **regression_metrics(predictions[TARGET], predictions[method])}
        for method in METHODS
    ]).sort_values(["MAE", "method"]).reset_index(drop=True)
    baseline = float(metrics.loc[metrics["method"].eq(METHODS[0]), "MAE"].iloc[0])
    metrics["MAE_improvement_vs_10game_pct"] = (
        100 * (baseline - metrics["MAE"]) / baseline if baseline else np.nan
    )
    return metrics
