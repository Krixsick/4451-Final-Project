# Blueprint v1 — verified validation results

These results use actual NBA data from the existing repository cache, transformed
offline through the shared preparation pipeline. Synthetic fixtures are used only
for tests. No 2026-27 test data were downloaded, opened, or evaluated.

## Dataset and protocol

- Training: 2021-22 through 2024-25, **83,461** eligible player appearances.
- Validation: 2025-26, **21,204** eligible player appearances.
- All comparisons use identical upstream complete cases and ten prior same-season appearances.
- 39 predeclared candidate fits; training-only fitting/scaling; fixed seeds and grids.
- Protocol SHA-256: `0f8cb68fa7ccb3327bac1917aa411df254ce6383bd3f3a49e24e627060b6dbed`.

Earlier exploratory results used an additional 2020-21 season and different
algorithms/settings. They are separate experiments and are not directly comparable.

## RQ1 — four official models on F3

| model_id | variant | n | MAE | RMSE |
| --- | --- | --- | --- | --- |
| M0 | M0_previous_10 | 21204 | 4.716219 | 6.181266 |
| M1 | M1_F3 | 21204 | 4.663954 | 6.038010 |
| M2 | M2_F3 | 21204 | 4.677721 | 6.063225 |
| M3 | M3_F3 | 21204 | 4.664943 | 6.045673 |


M0 is the validation-selected historical-average rule. Both baseline references
are retained in `baseline_reference.csv`. M1 has the lowest observed RQ1 MAE;
differences in this table do not establish statistical significance. MAE and RMSE
can rank candidates differently because RMSE emphasizes larger errors.

## RQ2 — all three feature groups for each ML family

| model_id | feature_set | n | MAE | RMSE | incremental_MAE_reduction |
| --- | --- | --- | --- | --- | --- |
| M1 | F1 | 21204 | 4.657306 | 6.049599 | — |
| M1 | F2 | 21204 | 4.666334 | 6.041010 | -0.009028 |
| M1 | F3 | 21204 | 4.663954 | 6.038010 | +0.002380 |
| M2 | F1 | 21204 | 4.677090 | 6.070877 | — |
| M2 | F2 | 21204 | 4.679022 | 6.065300 | -0.001932 |
| M2 | F3 | 21204 | 4.677721 | 6.063225 | +0.001301 |
| M3 | F1 | 21204 | 4.665225 | 6.053700 | — |
| M3 | F2 | 21204 | 4.667453 | 6.046399 | -0.002228 |
| M3 | F3 | 21204 | 4.664943 | 6.045673 | +0.002510 |


F1/F2/F3 contain 10/12/18 inputs. The last column is previous-group MAE minus
current-group MAE; positive values indicate improvement. Each combination is
independently validation-tuned under its family-specific grid. Extra context
does not consistently reduce error. These are descriptive validation findings.

## Locked validation candidate and export

Across both M0 rules and all nine ML variants, the selected candidate is
**M1_F1**, with validation MAE **4.657306** and
RMSE **6.049599** points. It reduces MAE by **0.058913** points
(1.25%) relative to the selected M0 rule on the same rows.
The candidate has the lowest validation MAE; this is not a final test claim.

Selected parameters: `{"alpha": 100.0, "solver": "svd"}`.
The final artifact was fitted on training rows only, without retuning or fitting
validation rows. Its feature-specific contract and runtime versions are recorded
in `model_manifest.json`.

- Maximum absolute difference against frozen validation predictions: 0.0.
- Maximum absolute difference after reload: 0.0.
- Independent CLI rows checked with the target removed and headers reversed: 21204.
- Maximum CLI difference from frozen validation predictions: 0.0.
- Focused PR verification: **20 tests passed**, covering data/leakage boundaries, experiments, notebook execution and artifact contracts. Local verification also reproduced all 21,204 target-free CLI predictions and executed the official notebook.
- Model SHA-256: `36812cfbf2ecb7d061c8fb0566a144f2b4b8fd7ac3b29f7c10be60cda6466e87`.

## Milestone status

M0-M3 and validation RQ1/RQ2 are complete. D2/D3 have reproducible validation-stage
artifacts. RQ3 remains pending the genuinely unseen 2026-27 test after candidate
lock. The IEEE report and GitHub development merge remain separate unfinished
deliverables. Earlier monthly validation checks are not RQ3 evidence.

Limits: predictions are conditional on player appearance, exclude early-season
and cold-start rows, and use a restricted complete-case cohort. Validation was
used for tuning and selection; no unseen-season reliability is claimed here.
