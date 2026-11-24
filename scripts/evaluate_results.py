"""Phase 4 Task C: Paired significance test, outlier audit, interpretability, and IEEE figures."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for WSL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.experiment_config import FEATURE_GROUPS, FIXED, SPLITS, SPLIT_FILES
from scripts.modeling import TARGET, load_clean_split
from scripts.prepare_data import PROJECT_ROOT

# Verified validation results from reports/BLUEPRINT_VALIDATION_RESULTS.md
RQ1_VAL = pd.DataFrame([
    {"model": "M0 (10-Game Avg)", "MAE": 4.716219, "RMSE": 6.181266},
    {"model": "M1 (Ridge F3)",    "MAE": 4.663954, "RMSE": 6.038010},
    {"model": "M2 (RF F3)",       "MAE": 4.677721, "RMSE": 6.063225},
    {"model": "M3 (XGBoost F3)",  "MAE": 4.664943, "RMSE": 6.045673},
])

RQ2_VAL = pd.DataFrame([
    {"model": "M1 (Ridge)",   "F1": 4.657306, "F2": 4.666334, "F3": 4.663954},
    {"model": "M2 (RF)",      "F1": 4.677090, "F2": 4.679022, "F3": 4.677721},
    {"model": "M3 (XGBoost)", "F1": 4.665225, "F2": 4.667453, "F3": 4.664943},
])

SOLID_BLUE = "#0044CC"
SOLID_RED = "#CC0000"
BLACK = "#000000"
FEATURES_F1 = FEATURE_GROUPS["F1"]


def paired_bootstrap_ci(y_true, pred_a, pred_b, n_boot=10000, seed=42):
    """Paired bootstrap 95% CI for (Model A - Model B) in MAE and RMSE.
    Negative values mean Model A (Ridge) has lower error than Model B (Baseline).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    abs_a = np.abs(pred_a - y_true)
    abs_b = np.abs(pred_b - y_true)
    sq_a = (pred_a - y_true) ** 2
    sq_b = (pred_b - y_true) ** 2

    idx = rng.integers(0, n, size=(n_boot, n))
    mae_diffs = abs_a[idx].mean(axis=1) - abs_b[idx].mean(axis=1)
    rmse_diffs = np.sqrt(sq_a[idx].mean(axis=1)) - np.sqrt(sq_b[idx].mean(axis=1))

    return {
        "n": int(n),
        "mae_diff_mean": float(abs_a.mean() - abs_b.mean()),
        "mae_diff_ci95": [float(np.percentile(mae_diffs, 2.5)), float(np.percentile(mae_diffs, 97.5))],
        "mae_p_value_two_sided": float(2 * min((mae_diffs >= 0).mean(), (mae_diffs <= 0).mean())),
        "rmse_diff_mean": float(np.sqrt(sq_a.mean()) - np.sqrt(sq_b.mean())),
        "rmse_diff_ci95": [float(np.percentile(rmse_diffs, 2.5)), float(np.percentile(rmse_diffs, 97.5))],
        "rmse_p_value_two_sided": float(2 * min((rmse_diffs >= 0).mean(), (rmse_diffs <= 0).mean())),
    }


def extract_ridge_coefficients(train_df):
    """Refit locked M1_F1 on training split only to inspect standardized weights."""
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=100.0, **FIXED["M1"])),
    ])
    with threadpool_limits(limits=2):
        model.fit(train_df[FEATURES_F1], train_df[TARGET])

    coefs = model.named_steps["ridge"].coef_
    return (
        pd.DataFrame({"feature": FEATURES_F1, "coef": coefs})
        .sort_values("coef", ascending=True)
        .reset_index(drop=True)
    )


def make_figures(preds, strat, train_df, val_df, coef_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    save_kwargs = {"dpi": 300, "facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.15}

    # --- Fig 1: RQ1 Model Comparison (Validation) ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(RQ1_VAL))
    w = 0.35
    ax.bar(x - w / 2, RQ1_VAL["MAE"], w, label="MAE", color=SOLID_BLUE, edgecolor=BLACK, linewidth=0.7)
    ax.bar(x + w / 2, RQ1_VAL["RMSE"], w, label="RMSE", color=SOLID_RED, edgecolor=BLACK, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(RQ1_VAL["model"], rotation=12, ha="right")
    ax.set_ylabel("Error (Points)")
    ax.set_title("RQ1: Validation Error Across Model Families\n(2025–26, n=21,204)")
    ax.set_ylim(4.0, 6.4)
    ax.legend(frameon=True, facecolor="white", edgecolor=BLACK)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_rq1_model_comparison.png", **save_kwargs)
    plt.close(fig)

    # --- Fig 2: RQ2 Feature Ablation (F1 -> F2 -> F3) ---
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sets = ["F1", "F2", "F3"]
    colors = [BLACK, SOLID_BLUE, SOLID_RED]
    for (_, row), color in zip(RQ2_VAL.iterrows(), colors):
        ax.plot(sets, [row["F1"], row["F2"], row["F3"]], marker="o", linewidth=2.2, label=row["model"], color=color)
    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Validation MAE (Points)")
    ax.set_title("RQ2: Feature Ablation\n(History vs. +Opponent vs. +Game Context)")
    ax.legend(frameon=True, facecolor="white", edgecolor=BLACK)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_rq2_feature_ablation.png", **save_kwargs)
    plt.close(fig)

    # --- Fig 3: RQ3 Predicted vs. Actual Scatter (2026-27 Holdout) ---
    fig, ax = plt.subplots(figsize=(6, 5.8))
    top10_idx = preds["ridge_points_error"].abs().nlargest(10).index
    normal = preds.drop(index=top10_idx)
    outliers = preds.loc[top10_idx]

    lim = max(preds["target_points"].max(), preds["ridge_points"].max()) + 2
    ax.scatter(normal["target_points"], normal["ridge_points"], alpha=0.55, s=18, color=SOLID_BLUE, edgecolors="none", label="2026–27 Appearances")
    ax.scatter(outliers["target_points"], outliers["ridge_points"], alpha=1.0, s=32, color=SOLID_RED, edgecolors=BLACK, linewidth=0.6, label="Top 10 Largest Misses")
    ax.plot([0, lim], [0, lim], color=BLACK, linestyle="--", linewidth=1.4, label="Perfect Prediction (y = x)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Points Scored")
    ax.set_ylabel("Ridge (M1_F1) Predicted Points")
    ax.set_title(f"RQ3: Predicted vs. Actual Points\n(2026–27 Holdout, n={len(preds):,})")
    ax.legend(frameon=True, facecolor="white", edgecolor=BLACK)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_rq3_pred_vs_actual.png", **save_kwargs)
    plt.close(fig)

    # --- Fig 4: RQ3 Error Distribution & Stratified Minutes MAE ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: Signed error histogram
    axes[0].hist(preds["ridge_points_error"], bins=35, alpha=0.75, color=SOLID_BLUE, edgecolor=BLACK, linewidth=0.4, label="Ridge F1")
    axes[0].hist(preds["baseline_points_error"], bins=35, alpha=0.55, color=SOLID_RED, edgecolor=BLACK, linewidth=0.4, label="10-Game Avg")
    axes[0].axvline(0, color=BLACK, linestyle="--", linewidth=1.3)
    axes[0].set_xlabel("Signed Error (Predicted - Actual Points)")
    axes[0].set_ylabel("Appearances")
    axes[0].set_title("RQ3: Signed Prediction Error\nDistribution")
    axes[0].legend(frameon=True, facecolor="white", edgecolor=BLACK)

    # Right: Stratified MAE by prior 10-game minutes
    mins = strat[strat["group"] == "prior_10_minutes"]
    order = ["under 20", "20 to under 30", "30 or more"]
    r_vals = [mins[(mins["value"] == v) & (mins["method"] == "Ridge F1")]["MAE"].values[0] for v in order]
    b_vals = [mins[(mins["value"] == v) & (mins["method"] == "Ten-game mean")]["MAE"].values[0] for v in order]
    x = np.arange(len(order))
    w = 0.35
    axes[1].bar(x - w / 2, r_vals, w, label="Ridge F1", color=SOLID_BLUE, edgecolor=BLACK, linewidth=0.7)
    axes[1].bar(x + w / 2, b_vals, w, label="10-Game Avg", color=SOLID_RED, edgecolor=BLACK, linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_ylabel("MAE (Points)")
    axes[1].set_title("RQ3: Held-Out MAE by\nPrior 10-Game Minutes")
    axes[1].legend(frameon=True, facecolor="white", edgecolor=BLACK)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_rq3_errors_and_strata.png", **save_kwargs)
    plt.close(fig)

    # --- Fig 5: Standardized Ridge (M1_F1) Coefficients ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = [f.replace("_", " ") for f in coef_df["feature"]]
    vals = coef_df["coef"].to_numpy()
    bar_colors = [SOLID_BLUE if v >= 0 else SOLID_RED for v in vals]

    y = np.arange(len(labels))
    ax.barh(y, vals, color=bar_colors, edgecolor=BLACK, linewidth=0.7)
    ax.axvline(0, color=BLACK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardized Coefficient (Points per 1 SD)")
    ax.set_title("M1_F1 Interpretability: Standardized Ridge Weights\n(Fitted on 2021–25 Training Split, alpha=100)")

    max_abs = np.max(np.abs(vals))
    pad_left = min(0.0, vals.min()) - 0.18 * max_abs
    pad_right = max(0.0, vals.max()) + 0.22 * max_abs
    ax.set_xlim(pad_left, pad_right)

    for idx_i, val in enumerate(vals):
        offset = 0.02 * max_abs if val >= 0 else -0.02 * max_abs
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, idx_i, f"{val:+.2f}", va="center", ha=ha, fontsize=9, color=BLACK)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_ridge_f1_coefficients.png", **save_kwargs)
    plt.close(fig)

    # --- Fig 6: Target Scoring Distribution Across Splits ---
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.arange(0, 62, 2)

    train_pts = train_df[TARGET].to_numpy()
    val_pts = val_df[TARGET].to_numpy()
    test_pts = preds["target_points"].to_numpy()

    ax.hist(
        train_pts, bins=bins, density=True, histtype="step",
        color=BLACK, linewidth=1.8, linestyle="--",
        label=f"Train 2021–25 (mean={train_pts.mean():.2f}, n={len(train_pts):,})",
    )
    ax.hist(
        val_pts, bins=bins, density=True, histtype="step",
        color=SOLID_BLUE, linewidth=2.0,
        label=f"Validation 2025–26 (mean={val_pts.mean():.2f}, n={len(val_pts):,})",
    )
    ax.hist(
        test_pts, bins=bins, density=True, histtype="stepfilled",
        color=SOLID_RED, alpha=0.14,
    )
    ax.hist(
        test_pts, bins=bins, density=True, histtype="step",
        color=SOLID_RED, linewidth=2.2,
        label=f"Holdout 2026–27 (mean={test_pts.mean():.2f}, n={len(test_pts):,})",
    )

    ax.set_xlim(0, 60)
    ax.set_xlabel("Actual Points Scored")
    ax.set_ylabel("Probability Density")
    ax.set_title("Target Scoring Distribution Across Chronological Splits\n(Train vs. Validation vs. 2026–27 Holdout)")
    ax.legend(frameon=True, facecolor="white", edgecolor=BLACK)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_target_scoring_distribution.png", **save_kwargs)
    plt.close(fig)


def main():
    clean = PROJECT_ROOT / "data/clean/blueprint_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=clean / SPLIT_FILES["train"])
    parser.add_argument("--validation-csv", type=Path, default=clean / SPLIT_FILES["validation"])
    parser.add_argument(
        "--rq3-dir", type=Path,
        default=PROJECT_ROOT / "artifacts/blueprint_v1/rq3_reproduction",
        help="Directory containing evaluate_rq3.py outputs",
    )
    parser.add_argument(
        "--fig-dir", type=Path,
        default=PROJECT_ROOT / "reports/figures",
        help="Output directory for IEEE figures",
    )
    args = parser.parse_args()

    train_df = load_clean_split(args.train_csv, SPLITS["train"])
    val_df = load_clean_split(args.validation_csv, SPLITS["validation"])
    preds = pd.read_csv(args.rq3_dir / "predictions.csv")
    strat = pd.read_csv(args.rq3_dir / "stratified_metrics.csv")

    # 1. Paired Bootstrap Significance Test
    stats = paired_bootstrap_ci(
        preds["target_points"].to_numpy(),
        preds["ridge_points"].to_numpy(),
        preds["baseline_points"].to_numpy(),
    )
    print("\n=== RQ3 Paired Bootstrap (Ridge F1 minus 10-Game Baseline) ===")
    print(json.dumps(stats, indent=2))

    # 2. Top 10 Largest Outlier Misses
    preds["abs_ridge_error"] = preds["ridge_points_error"].abs()
    cols = ["game_date", "player_name", "team_abbreviation", "target_points", "ridge_points", "ridge_points_error"]
    outliers = preds.sort_values("abs_ridge_error", ascending=False).head(10)[cols]
    print("\n=== Top 10 Largest Ridge Misses (2026–27 Holdout) ===")
    print(outliers.to_string(index=False))

    # 3. Standardized Ridge F1 Coefficients
    coef_df = extract_ridge_coefficients(train_df)
    print("\n=== Standardized Ridge (M1_F1) Coefficients ===")
    print(coef_df.sort_values("coef", ascending=False).to_string(index=False))

    # 4. Target Scoring Summary Across Splits
    split_summary = pd.DataFrame([
        {
            "split": name,
            "n": len(series),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "p90": series.quantile(0.90),
            "max": series.max(),
        }
        for name, series in [
            ("Train (2021–25)", train_df[TARGET]),
            ("Validation (2025–26)", val_df[TARGET]),
            ("Holdout (2026–27)", preds["target_points"]),
        ]
    ])
    print("\n=== Target Scoring Summary Across Splits ===")
    print(split_summary.to_string(index=False))

    # Save summary artifacts
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    (args.fig_dir / "rq3_bootstrap_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    outliers.to_csv(args.fig_dir / "rq3_top10_outliers.csv", index=False)
    coef_df.sort_values("coef", ascending=False).to_csv(args.fig_dir / "ridge_f1_coefficients.csv", index=False)
    split_summary.to_csv(args.fig_dir / "split_scoring_summary.csv", index=False)

    # 5. Export All 6 IEEE Figures
    make_figures(preds, strat, train_df, val_df, coef_df, args.fig_dir)
    print(f"\nSaved 6 IEEE figures + stats/tables to {args.fig_dir}")


if __name__ == "__main__":
    main()