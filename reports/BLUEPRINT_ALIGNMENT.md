# Official Blueprint PR scope and alignment

This change adds the official modeling/evaluation layer to the existing team
repository. The upstream nba_source.py, features.py, prepare_data.py and the
teammate's test_leakage.py are retained unchanged.

| Milestone | Included | Remaining |
| --- | --- | --- |
| D1 integration | Same raw cache, lagged inputs, snapshot/GP cutoffs and reports | Reconcile teammate-local snapshots when different files are used |
| M0/M1 | Historical means and scaled Ridge, common scoring | Final held-out evaluation |
| M2/M3 and RQ1 | Random Forest and actual XGBoost, F3 comparison | Final held-out evaluation |
| RQ2 | Nine nested-group combinations on identical rows | Interpretation in final report |
| D2/D3 | Validation engine, figures/manifests, export and inference | Final-test artifacts |
| RQ3 | Test remains unopened | Completed 2026-27 cutoff, coverage and errors |
| D4 | Detailed Methods walkthrough and validation findings | Related work, final Discussion/Conclusions and IEEE assembly |

Official training is 2021-22 through 2024-25, validation 2025-26. The legacy
prepare_data CLI still starts in 2020-21; official commands use scripts.blueprint
and data/clean/blueprint_v1. The protocol is in scripts/experiment_config.py.

The PR includes one modular notebook. The individual-download NBA_Main.ipynb
embeds fixed helpers and is not the repository entrypoint. Earlier experimental
notebooks/scripts are excluded from this focused change; their historical work
remains preserved separately. Scores from different protocols are not mixed.

Tracked content is source, the output-free notebook, compact actual-data example,
tests, dependency/ignore updates and reports. The local-test ZIP additionally
contains ignored generated clean inputs, validation results, a trusted frozen
model and a package-only verification runner. These are not in the source diff.

All training uses training rows only; MAE selects the winner and RMSE is secondary.
Every group retains the upstream full-input complete-case cohort. There is no
test-driven selection, validation refit, DNP prediction or in-game forecasting.
