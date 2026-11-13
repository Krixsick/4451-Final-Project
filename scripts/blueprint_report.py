"""Verify saved validation evidence and render the team's RQ1/RQ2 figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experiment_config import MODEL_LABELS, SPLITS, SPLIT_FILES, protocol_sha256
from scripts.modeling import TARGET, file_sha256, load_clean_split, regression_metrics


def read_verified_results(clean_dir, results_dir):
    clean_dir, results_dir = Path(clean_dir), Path(results_dir)
    manifest = json.loads((results_dir / "experiment_manifest.json").read_text())
    if manifest.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Saved results use a different official protocol")
    if manifest.get("final_test_loaded") or manifest.get("final_test_evaluated"):
        raise ValueError("These are validation-only results")
    for row in manifest["inputs"]:
        if row["split"] not in ["train", "validation"]:
            raise ValueError("Unexpected experiment input")
        for filename, digest in [(row["filename"], row["sha256"]), (row["report"], row["report_sha256"])]:
            if file_sha256(clean_dir / filename) != digest:
                raise ValueError("Saved results do not match the supplied data/reports")
    tables = {name: pd.read_csv(results_dir / f"{name}.csv") for name in
        ["rq1_model_comparison", "rq2_feature_ablation", "baseline_reference", "validation_all_variants",
         "candidate_scores", "preparation_coverage"]}
    predictions = pd.read_csv(results_dir / "validation_predictions.csv", dtype={"game_id": "string"})
    validation = load_clean_split(clean_dir / SPLIT_FILES["validation"], SPLITS["validation"])
    predictions["game_date"] = pd.to_datetime(predictions["game_date"]).dt.normalize()
    pd.testing.assert_frame_equal(predictions[["game_id", "player_id", "game_date", TARGET]],
        validation[["game_id", "player_id", "game_date", TARGET]], check_dtype=False)
    for row in tables["validation_all_variants"].to_dict("records"):
        actual = regression_metrics(predictions[TARGET], predictions[row["variant"]])
        np.testing.assert_allclose([actual["MAE"], actual["RMSE"], actual["n"]],
            [row["MAE"], row["RMSE"], row["n"]], rtol=1e-10)
    tables.update(manifest=manifest, predictions=predictions)
    return tables


def render_figures(tables, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rq1, rq2, predictions = (tables[key] for key in ["rq1_model_comparison", "rq2_feature_ablation", "predictions"])
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    short_names = {"M0": "Average", "M1": "Ridge", "M2": "Random Forest", "M3": "XGBoost"}
    labels = [f"{row.model_id}\n{short_names[row.model_id]}" for row in rq1.itertuples()]
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        bars = ax.bar(labels, rq1[metric], color=["#788897", "#4285aa", "#72a478", "#c38b43"])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set(ylabel=f"{metric} (points)", title=f"RQ1 — validation {metric}")
        ax.set_ylim(0, rq1[metric].max() * 1.18)
    fig.savefig(output_dir / "rq1_model_metrics.png", dpi=170)
    figures.append(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, rq1.itertuples()):
        ax.hexbin(predictions[TARGET], predictions[row.variant], gridsize=45, mincnt=1, cmap="Blues")
        limit = max(predictions[TARGET].max(), predictions[row.variant].max()) + 2
        ax.plot([0, limit], [0, limit], color="#ad694f", linestyle="--", linewidth=1)
        ax.set(xlabel="Actual points", ylabel="Predicted points", title=f"{row.model_id}: {MODEL_LABELS[row.model_id]}")
    fig.savefig(output_dir / "rq1_actual_vs_predicted.png", dpi=170)
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model_id, rows in rq2.groupby("model_id", sort=True):
        for ax, metric in zip(axes, ["MAE", "RMSE"]):
            ax.plot(rows["feature_set"], rows[metric], marker="o", label=model_id)
            ax.set(xlabel="Nested feature set", ylabel=f"{metric} (points)", title=f"RQ2 — validation {metric}")
            ax.legend()
    fig.savefig(output_dir / "rq2_feature_ablation.png", dpi=170)
    figures.append(fig)
    return figures
