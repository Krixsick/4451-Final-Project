"""Official team preparation and RQ1/RQ2 experiments; RQ3 stays held out."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from scripts.experiment_config import (PROTOCOL_ID, SPLITS, SPLIT_FILES, FEATURE_GROUPS,
    MODEL_LABELS, FIXED, CANDIDATES, protocol_spec, protocol_sha256)
from scripts.features import FEATURE_COLUMNS
from scripts.model_artifact import runtime_versions
from scripts.modeling import METADATA_COLUMNS, TARGET, file_sha256, load_clean_split, preparation_report, regression_metrics, validate_split_order
from scripts.nba_source import NBAData, save_csv
from scripts.prepare_data import PROJECT_ROOT, prepare_season


def prepare_blueprint(data_dir, output_dir, offline=False):
    """Reuse the original extraction/transformation functions with explicit official splits."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new official data directory; previous experiments are preserved")
    source = NBAData(Path(data_dir) / "dirty", offline=offline)
    tables, reports = {}, {}
    for season in SPLITS["train"] + SPLITS["validation"]:
        tables[season], reports[season] = prepare_season(season, source, output_dir)
    for split in ["train", "validation"]:
        combined = pd.concat([tables[season] for season in SPLITS[split]], ignore_index=True)
        if combined.empty or combined.duplicated(["game_id", "player_id"]).any():
            raise ValueError("Invalid combined official split")
        save_csv(combined, output_dir / SPLIT_FILES[split])
        report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": PROTOCOL_ID,
            "feature_columns": FEATURE_COLUMNS, "target_column": TARGET, "player_ids": None,
            "split_seasons": {split: SPLITS[split]}, "seasons": [reports[s] for s in SPLITS[split]],
            "missing_value_policy": "Unchanged upstream complete-case policy"}
        (output_dir / f"report_{split}.json").write_text(json.dumps(report, indent=2))
    (output_dir / "protocol.json").write_text(json.dumps(protocol_spec(), indent=2))


def build_estimator(model_id, candidate):
    params = {**FIXED[model_id], **candidate}
    if model_id == "M1":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(**params))])
    if model_id == "M2":
        return RandomForestRegressor(**params)
    if model_id == "M3":
        return XGBRegressor(**params)
    raise ValueError("Unknown official model ID")


def fit_matrix(train, validation, candidate_grids=None):
    validate_split_order(train, validation)
    grids = CANDIDATES if candidate_grids is None else candidate_grids
    if set(grids) != set(CANDIDATES) or any(not grids[key] for key in grids):
        raise ValueError("Provide candidate grids for M1, M2, and M3")
    predictions = validation[METADATA_COLUMNS + [TARGET]].copy()
    rows, tuning, models, choices = [], [], {}, {}
    for variant, column in [("M0_previous_10", "points_last_10"), ("M0_season", "points_season_avg")]:
        predictions[variant] = validation[column]
        rows.append({"variant": variant, "model_id": "M0", "feature_set": "baseline", "feature_count": 2,
                     "source_feature": column, **regression_metrics(validation[TARGET], predictions[variant])})
    with threadpool_limits(limits=2):
        for model_id in ["M1", "M2", "M3"]:
            for group, features in FEATURE_GROUPS.items():
                variant = f"{model_id}_{group}"
                best, best_key, best_params = None, None, None
                for index, candidate in enumerate(grids[model_id]):
                    if set(candidate) != set(CANDIDATES[model_id][0]):
                        raise ValueError("Candidate keys disagree with the declared grid")
                    estimator = build_estimator(model_id, candidate)
                    estimator.fit(train[features], train[TARGET])
                    predicted = np.asarray(estimator.predict(validation[features]), dtype=float)
                    score = regression_metrics(validation[TARGET], predicted)
                    tuning.append({"variant": variant, "model_id": model_id, "feature_set": group,
                        "candidate": index, "candidate_parameters": json.dumps(candidate, sort_keys=True), **score})
                    key = (score["MAE"], index)
                    if best_key is None or key < best_key:
                        best, best_key, best_params = estimator, key, candidate
                    else:
                        del estimator
                predicted = np.asarray(best.predict(validation[features]), dtype=float)
                predictions[variant] = predicted
                rows.append({"variant": variant, "model_id": model_id, "feature_set": group,
                             "feature_count": len(features), **regression_metrics(validation[TARGET], predicted)})
                models[variant] = best
                choices[variant] = {**FIXED[model_id], **best_params}
                print(f"{variant}: MAE={best_key[0]:.6f}", flush=True)
                gc.collect()
    scores = pd.DataFrame(rows)
    baseline = scores.loc[scores["model_id"].eq("M0")].sort_values(["MAE", "variant"]).iloc[0]
    rq1 = scores.loc[scores["variant"].eq(baseline["variant"]) | scores["feature_set"].eq("F3")].copy()
    rq1 = rq1.sort_values("model_id").reset_index(drop=True)
    rq2 = scores.loc[scores["model_id"].ne("M0")].sort_values(["model_id", "feature_set"]).copy()
    rq2["incremental_MAE_reduction"] = -rq2.groupby("model_id")["MAE"].diff()
    rq2["previous_feature_set"] = rq2.groupby("model_id")["feature_set"].shift(1)
    selected = scores.sort_values(["MAE", "feature_count", "model_id", "feature_set", "variant"]).iloc[0]
    return {"scores": scores, "rq1": rq1, "rq2": rq2, "baseline": scores.loc[scores.model_id.eq("M0")],
        "predictions": predictions, "tuning": pd.DataFrame(tuning), "models": models,
        "parameters": choices, "selected": selected.dropna().to_dict()}


def run_blueprint(clean_dir, output_dir, candidate_grids=None):
    clean_dir, output_dir = Path(clean_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a new official results directory")
    frames, reports, coverage, inputs = {}, {}, [], []
    for split in ["train", "validation"]:
        path = clean_dir / SPLIT_FILES[split]
        frame = load_clean_split(path, SPLITS[split])
        report, table, report_path = preparation_report(clean_dir, split, frame, SPLITS[split])
        if report is None:
            raise ValueError("Official preparation report is required")
        frames[split], reports[split] = frame, report
        coverage.append(table)
        inputs.append({"split": split, "filename": path.name, "sha256": file_sha256(path), "rows": len(frame),
                       "report": report_path.name, "report_sha256": file_sha256(report_path)})
    result = fit_matrix(frames["train"], frames["validation"], candidate_grids)
    result["coverage"] = pd.concat(coverage, ignore_index=True)
    result["manifest"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha256(), "protocol": protocol_spec(),
        "executed_candidate_grids": CANDIDATES if candidate_grids is None else candidate_grids,
        "is_full_protocol_grid": candidate_grids is None or candidate_grids == CANDIDATES,
        "inputs": inputs, "preparation_reports": reports, "selected_candidate": result["selected"],
        "selected_parameters": result["parameters"],
        "validation_scores": [{k: v for k, v in row.items() if pd.notna(v)}
                              for row in result["scores"].to_dict(orient="records")],
        "runtime_versions": runtime_versions(),
        "final_test_loaded": False, "final_test_evaluated": False, "rq3_status": "pending_unseen_test",
        "code_sha256": {name: file_sha256(PROJECT_ROOT / "scripts" / name)
                        for name in ["blueprint.py", "experiment_config.py", "features.py"]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {"rq1_model_comparison": result["rq1"], "rq2_feature_ablation": result["rq2"],
        "baseline_reference": result["baseline"], "validation_all_variants": result["scores"],
        "candidate_scores": result["tuning"], "validation_predictions": result["predictions"],
        "preparation_coverage": result["coverage"]}.items():
        save_csv(table, output_dir / f"{name}.csv")
    (output_dir / "experiment_manifest.json").write_text(json.dumps(result["manifest"], indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    prepare.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/clean/blueprint_v1")
    prepare.add_argument("--offline", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--clean-dir", type=Path, default=PROJECT_ROOT / "data/clean/blueprint_v1")
    evaluate.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/blueprint_v1/results")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare_blueprint(args.data_dir, args.output_dir, args.offline)
    else:
        result = run_blueprint(args.clean_dir, args.output_dir)
        print(result["rq1"].to_string(index=False))
        print("Selected candidate:", result["selected"])


if __name__ == "__main__":
    main()
