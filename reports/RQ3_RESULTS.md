# RQ3 · Forecasting points on unseen 2026–27 NBA games

**Ridge retained a small scoring advantage on the held-out games.** Its predictions missed by 4.875 points per player appearance on average, compared with 4.922 for the preceding-ten-game scoring mean. The advantage was 0.047 points (0.95%) across 1,496 appearances. Both methods had larger errors than in validation.

## Results in points

| Period | Eligible appearances | Ridge MAE | Baseline MAE | Ridge RMSE | Baseline RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021–22 through 2024–25 training, in sample | 83,461 | 4.701 | 4.774 | 6.114 | 6.260 |
| 2025–26 validation | 21,204 | 4.657 | 4.716 | 6.050 | 6.181 |
| 2026–27 held-out window through Nov 22 | 1,496 | **4.875** | **4.922** | **6.404** | **6.493** |

From validation to the held-out window, Ridge MAE increased by 0.218 points and baseline MAE increased by 0.206. Ridge's RMSE increased by 0.354 points and the baseline's by 0.312. RMSE gives more weight to large misses. Ridge underestimated scoring by 0.332 points on average. Its median absolute miss was 4.00 points, its 90th percentile was 9.81, and its 95th percentile was 12.18. The baseline's corresponding absolute misses were 4.00, 9.90 and 12.40.

## Coverage and interpretation

The NBA logs cover 248 completed regular-season games from October 20 through November 22, 2026. Of 5,545 player appearances, 4,049 had fewer than ten previous appearances in the season. The 1,496 eligible appearances (27.0%) came from 307 players in 104 games between November 7 and November 22. Every scored player appeared in the game. Player box-score points summed to each team's final score, and team game counts matched the NBA's cumulative totals through the cutoff.

| Pregame grouping | Appearances | Ridge MAE | Baseline MAE |
| --- | ---: | ---: | ---: |
| Under 20 minutes per game over the previous 10 appearances | 442 | 3.913 | 3.962 |
| 20 to under 30 minutes | 646 | 5.040 | 5.083 |
| 30 minutes or more | 408 | 5.657 | 5.709 |
| Away | 771 | 4.821 | 4.829 |
| Home | 725 | 4.933 | 5.022 |
| Team playing a back-to-back | 279 | 4.427 | 4.489 |
| Team with at least one rest day | 1,217 | 4.978 | 5.022 |

Both methods had larger absolute misses for players who usually played more minutes. The group comparisons are descriptive. All eligible games fall in November, so this window cannot establish month-to-month stability. The small observed advantage applies to eligible appearances in this early-season window.

## Reproduction

The [evaluation protocol](RQ3_PROTOCOL.md) fixed the cutoff, Ridge settings, reference rule and metrics before opening the holdout. The packaged model binary was absent from this checkout. We reproduced the selected training-only fit from the official training data. Four player display-name spellings in the local 2024–25 history differed from the original export. Restoring them recovered the exact training CSV checksum. The validation CSV also matched Appendix C.4 of [the executed notebook](../notebooks/NBA_Blueprint_Experiments_Executed.ipynb). The reconstructed model reproduced the published validation MAE and RMSE on all 21,204 appearances. An independent solution of the Ridge equations agreed with all 1,496 held-out predictions to within 0.000000000001 points.

From the project root, the following commands use the cached data in this checkout. Omit `--offline` to fetch missing NBA records. With the team's prepared package, the evaluator's default training and validation paths point to the official split files.

```sh
.venv/bin/python -m scripts.prepare_rq3 --offline
.venv/bin/python -m scripts.evaluate_rq3 \
  --train-csv artifacts/blueprint_v1/rq3_inputs/official_train_2021_2025.csv \
  --validation-csv data/clean/validation_2025_2026.csv \
  --output-dir artifacts/blueprint_v1/rq3_reproduction
```

The original run's [predictions](../artifacts/blueprint_v1/rq3_through_2026-11-22/predictions.csv), [metrics](../artifacts/blueprint_v1/rq3_through_2026-11-22/metrics.csv), [error distributions](../artifacts/blueprint_v1/rq3_through_2026-11-22/error_distribution.csv), [group summaries](../artifacts/blueprint_v1/rq3_through_2026-11-22/stratified_metrics.csv) and [input checksums and coverage](../artifacts/blueprint_v1/rq3_through_2026-11-22/evaluation.json) are local generated outputs. The commands regenerate them in the specified results directory.
