# NBA player-points forecasting — team Blueprint

This PR adds modeling and validation evaluation to the existing NBA preparation
pipeline. One general model predicts points for eligible player appearances.

Start with [notebooks/NBA_Blueprint_Experiments.ipynb](notebooks/NBA_Blueprint_Experiments.ipynb).
The English notebook explains collection, cleaning, all 18 pregame inputs,
chronological splits, model/search definitions, results and inference. Read
[reports/BLUEPRINT_VALIDATION_RESULTS.md](reports/BLUEPRINT_VALIDATION_RESULTS.md)
for the recorded real-data findings.

## Setup and execution

Use Python 3.13, pinned by `.python-version`, and run from the project root:

```sh
uv sync --frozen --group dev
```

Select the project's `.venv` Python kernel in VS Code/Jupyter. The local-test ZIP
supplies data, verified results and a frozen model, so Run All works offline.
Its package-only `local_test/README.md` documents one-command verification.

A Git-only checkout needs prepared input files first:

```sh
uv run --frozen python -m scripts.blueprint prepare
uv run --frozen python -m scripts.blueprint evaluate
```

Preparation reuses the original NBA cache under `data/dirty`; add `--offline`
only when that cache is complete. It writes training/validation data to
`data/clean/blueprint_v1`, not the final test. Evaluation writes
`artifacts/blueprint_v1/results`. Existing outputs are preserved; choose fresh
output paths for an intentional new run. The notebook verifies and reuses saved
results when present, otherwise runs the full search. It also reproduces the
selected training-only fit.

## Official protocol

| Split | Exact seasons | Role |
| --- | --- | --- |
| Training | 2021-22, 2022-23, 2023-24, 2024-25 | Fit models and preprocessing |
| Validation | 2025-26 | Select parameters and the candidate |
| Reserved test | 2026-27 | Pending RQ3 |

`scripts/experiment_config.py` defines periods, nested inputs, grids and rules.
The unchanged legacy `prepare_data` CLI includes 2020-21 by default; use
`scripts.blueprint` for the official four-season protocol. Do not mix scores
from different training periods.

Each row contains eight metadata fields, 18 numeric inputs and `target_points`.
All variants retain the same upstream complete-case cohort and ten previous
same-season appearances. Names/IDs are metadata. Histories exclude the current
game, reset by season, continue across trades, and do not zero-fill absences.
Advanced summaries stop before the game and verify GP against prior team games.
Rest uses every team game, including games the selected player missed.

| Group | Input count | Addition |
| --- | --- | --- |
| F1 | 10 | Eight lagged player-stat fields, age and prior-appearance count |
| F2 | 12 | Opponent defensive rating and opponent pace |
| F3 | 18 | Own pace, home status, both rest intervals and both back-to-back flags |

## Models and evaluation

| ID | Procedure | Candidate configurations per group |
| --- | --- | --- |
| M0 | Previous-ten-game or season-to-date scoring mean | Two reference rules |
| M1 | StandardScaler then Ridge | Five alpha values |
| M2 | Random Forest, 150 trees | Four depth/minimum-leaf pairs |
| M3 | Actual XGBoost CPU histogram regressor, 300 trees | Four depth/L2 pairs |

There are 39 ML candidate fits across three groups. Scaling and fitting use
training only. Validation MAE selects parameters; RMSE is secondary. There is
no validation refit or early stopping. Seeds/grids are fixed before results.
Within-grid ties retain candidate order; overall ties prefer fewer inputs then
model/group/variant order.

RQ1 compares the selected M0 and M1/M2/M3 on F3. RQ2 evaluates all nine ML/group
combinations on identical rows; each is independently tuned. RQ3 remains pending.
The recorded candidate is Ridge F1, alpha 100: validation MAE 4.657306 versus
4.716219 for the ten-game mean. These development scores are not final-test claims.

## Export and target-free prediction

```sh
uv run --frozen python -m scripts.blueprint_artifact
uv run --frozen python -m scripts.predict --model-dir artifacts/blueprint_v1/frozen_model --input prepared_pregame.csv --output predictions.csv
```

Export verifies the recorded selection, data/report hashes, training-only fit,
validation reproduction and reload. Every declared winner is supported. XGBoost
uses native JSON, Ridge/Random Forest trusted joblib, and M0 a JSON rule.
Inference requires the selected pregame features, not the target. Headers map by
name; optional metadata and row order are preserved. No imputation, rounding or
clipping is applied. Load only trusted team artifacts with the locked environment.

The CLI does not implement availability prediction, automatic upcoming-game
feature assembly or FastAPI serving. The existing main.py is a placeholder.

## Tests and files

```sh
uv run --frozen --group dev python -m unittest discover -s tests -v
```

The focused suite preserves the teammate's rolling-history tests and checks
official contracts, leakage/snapshot boundaries, all-family serialization,
inference and notebook execution. Synthetic experiments use invented records
and a mocked NBA source. Notebook tests isolate data/results/model paths from
packaged real artifacts and select the current test process's Python kernel.

`examples/blueprint_walkthrough` is a separate actual NBA teaching subset:
Barnes's first 14 appearances in 2024-25, complete team history through the cutoff
and prior-day summaries. It reconstructs four eligible rows matching the official
season export. It is not the experiment training or validation cohort.

| Location | Role |
| --- | --- |
| scripts/nba_source.py, features.py, prepare_data.py | Unchanged collection/preparation |
| scripts/experiment_config.py, blueprint.py | Official protocol and experiment |
| scripts/modeling.py, blueprint_report.py | Contracts, scoring and figures |
| scripts/blueprint_artifact.py, model_artifact.py, predict.py | Export and inference |
| notebooks/NBA_Blueprint_Experiments.ipynb | Output-free official notebook |
| tests/, reports/ | Focused verification, scope and findings |

Generated `data/`, `artifacts/`, and package-only `local_test/` are ignored by Git.
The local-test ZIP supplies their required inputs/evidence separately. Earlier
exploratory work is outside this focused PR and remains preserved separately.
Branch from development, review the feature PR into development, and promote to
main after stable team integration. The individual-download NBA_Main.ipynb is a
separate distribution, not the shared repository entrypoint.

Final RQ3, the IEEE report and serving application remain unfinished milestones.
Document appearance conditioning, warm-up exclusions, complete-case coverage,
retrospective sources and validation-selection optimism in the report.
