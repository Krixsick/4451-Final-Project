# 4451-Final-Project

NBA player-points prediction case study. Prepare reproducible CSV datasets first;
train and evaluate models separately, then serve the chosen model through FastAPI.

## Where to start reading the code

The pipeline is organized into three modules:

| File | Responsibility |
| --- | --- |
| `scripts/prepare_data.py` | Run the steps and export the CSVs |
| `scripts/features.py` | Calculate player history, team rest, age, and opponent features |
| `scripts/nba_source.py` | Request NBA data, cache responses, and retry failed downloads |

Start with `prepare_season()` in `scripts/prepare_data.py`. It follows four steps:
load records, calculate features, separate incomplete rows, and save CSVs.
`add_player_history()` has one explicit assignment per rolling feature.
Team context is broken into schedule calculations, historical ratings, and
opponent matching. Download details live in their own module so you can learn
the feature calculations without first reading request and cache handling.

## Prepare the datasets

From the project root, start with one player and one season:

```sh
uv run python -m scripts.prepare_data --season 2024-25 --player-id 2544
```

The source downloads are cached under `data/dirty`. A small player sample is
exported under `data/clean/samples/players_2544`, so it cannot overwrite the
full-study files. Omit `--player-id` to include all players. No manual name-to-ID
lookup is required: the league-wide player logs contain both.

Build each study split:

```sh
uv run python -m scripts.prepare_data --split train
uv run python -m scripts.prepare_data --split validation
uv run python -m scripts.prepare_data --split test --refresh-logs
```

| Export | Seasons |
| --- | --- |
| `data/clean/train_2020_2025.csv` | 2020-21 through 2024-25 |
| `data/clean/validation_2025_2026.csv` | 2025-26 |
| `data/clean/test_2026_2027.csv` | Completed 2026-27 regular-season appearances available when downloaded |

`--split all` prepares all three. Keep final-test outcomes out of feature and
model selection. The ongoing test season is not a complete-season evaluation yet.

The first download can be slow: two league-wide game-log requests per season,
one profile per distinct player, and one league-wide advanced team snapshot per
game date. Cached files allow interrupted runs to resume. Each request has a
timeout, a bounded retry count, and spacing between requests. If NBA endpoints
are unavailable, the command fails explicitly; it never invents replacement data.

Other options:

- `--offline`: prepare CSVs using cached sources only.
- `--refresh-logs`: refresh player and team logs as new games finish.
- `--season 2025-26`: export that season without rebuilding combined splits.
- `--player-id 2544 --player-id 201939`: process a selected sample.
- `--data-dir /absolute/path`: override the data folder.
- `--timeout 30 --attempts 2`: control network retries.

Profiles and date-specific team snapshots stay cached. Refresh a corrected
snapshot by removing its specific cache file deliberately. Preserve frozen raw
data with the final experiment so the results remain reproducible.

## CSV contents

One row represents one player's appearance in one game. Metadata columns are
`season`, `game_id`, `game_date`, `player_id`, `player_name`, `team_id`,
`team_abbreviation`, and `opponent_id`. Names are retained for inspection and
reporting; IDs are the stable join keys. Names and IDs are not model inputs.

The 18 numeric input columns are defined in `scripts/features.py`:

| Group | Columns |
| --- | --- |
| Scoring | points_last_5, points_last_10, points_season_avg |
| Minutes | minutes_last_5, minutes_last_10 |
| Attempts | fga_last_10, fg3a_last_10, fta_last_10 |
| Opponent defense | opponent_defensive_rating |
| Pace | team_pace, opponent_pace |
| Schedule | is_home, team_rest_days, opponent_rest_days, team_back_to_back, opponent_back_to_back |
| Background | age, games_played_before |

`target_points` is the observed outcome, stored separately from the input list.

```python
import pandas as pd
from scripts.features import FEATURE_COLUMNS

training = pd.read_csv(
    "data/clean/train_2020_2025.csv", dtype={"game_id": "string"}
)
X_train = training[FEATURE_COLUMNS]
y_train = training["target_points"]
names_for_reporting = training["player_name"]
```

Use the explicit input list rather than automatically feeding every CSV column
into a model. Fit any scaler, imputer, or other learned preprocessing on training
data only. These CSVs are not scaled and are not used to train anything by the
preparation command.

## Cleaning and historical cutoffs

- Historical games are sorted before calculations. Rolling averages and season
  averages exclude the current appearance with `shift(1)`.
- Histories reset at a season boundary but remain continuous across a trade.
  This initial version does not use a prior-season fallback.
- A ten-game average requires ten earlier appearances. An absence is not entered
  as a zero-point game. The study is conditional on the player making an appearance.
- Complete rows go to the clean CSVs. Rows missing any input go to
  `data/clean/excluded/excluded_<season>.csv` with `missing_columns`. No missing
  value is silently turned into zero. This policy excludes early-season and
  low-history appearances; document that population in the study results.
- The proposed 70% attendance rule is not applied. It needs a separate prior-season
  eligibility decision and must not use the eventual attendance of a test season.
- Rest uses every team game, including games a selected player missed. Rest means
  full calendar days between games: consecutive dates mean 0 rest and a back-to-back.
  First-game rest is unknown because these logs contain regular-season games only.
- Opponent rating and both pace values are NBA-returned advanced statistics with
  `DateTo` set to the previous calendar day. The pipeline checks the returned GP
  against the number of earlier team games to detect a missing/ignored cutoff.
- First-game team ratings stay missing. Team identity comes from the historical
  game, not today's roster. Age is fractional years since birth on the game date.
- Duplicate player-game records, missing names, missing birthdates, invalid
  historical statistics, inconsistent team-game joins, and infinite values fail
  explicitly. No duplicate is arbitrarily chosen or averaged away.
- Each run writes a JSON report with split membership, feature names, source/clean
  row counts, exclusions, and last game date. This date identifies the extent of
  an in-progress season, not a guarantee that a full season was returned.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Tests use clearly invented records and mocked requests. They cover rolling
arithmetic, future-data exclusion, trades, team rest across player absences,
Unicode names, leading-zero game IDs, validation/test separation, cache reuse,
cutoff verification, and excluded-row reports. They do not establish NBA server
availability. Never use the synthetic fixtures as study data.

## Source documentation

- [Player and team game logs](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md)
- [Advanced team statistics and date filters](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguedashteamstats.md)
- [Player birthdates](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/commonplayerinfo.md)

## Train the first points model

After preparing the training and validation exports, run:

```sh
uv run python -m scripts.train_model
```

Read `scripts/train_model.py` starting at `train_and_evaluate()`. It selects the
18 `FEATURE_COLUMNS`, fits one shared gradient-boosted tree regressor, and
predicts points for the later validation games. Names and IDs remain in the
prediction report but never enter the model. There is no random train/test
split, and automatic early stopping is disabled to avoid an internal random
validation split. The fixed settings are a starting experiment, not tuned values.
Predictions are clipped to a minimum of zero.

The default inputs are `data/clean/train_2020_2025.csv` and
`data/clean/validation_2025_2026.csv`. Missing files cause an error: the script
does not silently substitute partial data or download NBA records.

To try the code with the already prepared earlier seasons:

```sh
uv run python -m scripts.train_model \
  --train data/clean/seasons/features_2020-21.csv \
  --validation data/clean/seasons/features_2021-22.csv \
  --output data/models/early_experiment
```

That is an exploratory 2020-21 -> 2021-22 experiment, not the planned five-season
training / 2025-26 validation result. All training dates must precede every
validation date, and duplicate/overlapping player-game records are rejected.

By default, `data/models/model_a` receives:

- `model.pkl`: fitted model, ordered feature names, prediction floor, and scikit-learn version.
- `metrics.csv`: MAE and RMSE for the model, season-to-date average, and last-ten-game average.
- `validation_predictions.csv`: those three predictions and actual points for each player-game, with names.
- `report.json`: source paths, season/date coverage, row counts, model settings, and scores.

Lower MAE/RMSE is better. All three methods use exactly the same validation rows.
Features update using completed earlier games during validation, while model
parameters stay fixed. This simulates successive pregame predictions, not a
forecast of every game made before the season starts. Scores describe recorded
appearances with sufficient history; they do not evaluate DNP predictions.

Rerunning replaces results in the chosen output folder. Use a different
`--output` folder for each feature experiment. Only load pickle models you trust
and use the recorded scikit-learn environment when reloading. Training runs
outside FastAPI; the saved model can later be loaded for serving. Keep 2026-27
out of feature/model selection and evaluate it once choices are frozen.

Model reference: [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
