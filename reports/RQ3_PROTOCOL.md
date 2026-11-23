# RQ3 held-out evaluation

Run the prespecified forecast on completed NBA regular-season games from the start of 2026–27 through **November 22, 2026**, inclusive by NBA game date. This is the last full calendar day before the November 23 evaluation. All conclusions apply to this partial-season window.

## Decisions fixed before opening the holdout

- Use the validation winner `M1_F1` from Blueprint v1. Standardize the 10 F1 inputs and apply Ridge regression with `alpha=100` and the SVD solver. Use the fit from the four training seasons 2021–22 through 2024–25. Leave the fitted parameters fixed for the entire test.
- The executed notebook records the SHA-256 of the 83,461-row training CSV as `8dd8238d58def16ef522e55c0b648f4bbc1619830e0dda7c10329639499b6cdb`. If the packaged binary is unavailable, reproduce the same deterministic fit from that exact CSV before scoring. The recorded historical reference is each player's mean points over the preceding ten appearances.
- Use the shared league-wide, 18-input complete-case preparation, even though Ridge consumes only F1. Require ten earlier appearances in the same season. Use the full completed team schedule to compute rest and prior-day advanced snapshots to check team game counts. Earlier completed games in the test window can inform subsequent pregame histories.
- Score the frozen Ridge and the ten-game reference on identical appearances. Report source, eligible and excluded appearances, distinct games and players, actual date coverage, MAE, RMSE and validation-to-test changes. Summarize signed and absolute error, and report scores by month, prior ten-game minutes (under 20, 20 to under 30, 30 or more), home/away, and team back-to-back. Include counts for interpreting small groups. Keep the selection fixed regardless of the results.

The validation reference values are in [the validation report](BLUEPRINT_VALIDATION_RESULTS.md). The training data identity is in Appendix C.4 of [the executed notebook](../notebooks/NBA_Blueprint_Experiments_Executed.ipynb).
