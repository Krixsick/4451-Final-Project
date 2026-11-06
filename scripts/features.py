"""Pregame features computed without current-game outcomes."""

import pandas as pd


FEATURE_COLUMNS = [
    "points_last_5", "points_last_10", "points_season_avg",
    "minutes_last_5", "minutes_last_10",
    "fga_last_10", "fg3a_last_10", "fta_last_10",
    "opponent_defensive_rating", "team_pace", "opponent_pace",
    "is_home", "team_rest_days", "opponent_rest_days",
    "team_back_to_back", "opponent_back_to_back",
    "age", "games_played_before",
]


def normalize_games(frame):
    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"]).dt.normalize()
    frame["GAME_ID"] = frame["GAME_ID"].astype("string")
    if frame[["season", "GAME_ID", "GAME_DATE", "TEAM_ID"]].isna().any().any():
        raise ValueError("Missing game identifiers or dates")
    return frame


def previous_average(values, window):
    """Average previous appearances; shift excludes the current game."""
    return values.shift(1).rolling(window, min_periods=window).mean()


def previous_season_average(values):
    """Average every earlier appearance in this season."""
    return values.shift(1).expanding().mean()


def add_player_history(player_games, birthdates):
    """Reset histories by season; trades do not reset a player's history."""
    games = normalize_games(player_games)
    if games[["PLAYER_ID", "PLAYER_NAME"]].isna().any().any():
        raise ValueError("Every player appearance needs an ID and name")
    if games["PLAYER_NAME"].astype(str).str.strip().eq("").any():
        raise ValueError("Player names must not be empty")
    for column in ["PTS", "MIN", "FGA", "FG3A", "FTA"]:
        games[column] = pd.to_numeric(games[column], errors="raise")
        if games[column].isna().any() or (games[column] < 0).any():
            raise ValueError(f"Missing or negative historical values in {column}")
    if games.duplicated(["season", "GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate player-game rows")
    games = games.sort_values(["PLAYER_ID", "season", "GAME_DATE", "GAME_ID"])
    groups = games.groupby(["PLAYER_ID", "season"], sort=False)

    # Each line corresponds directly to a column in the cleaned CSV.
    games["points_last_5"] = groups["PTS"].transform(previous_average, window=5)
    games["points_last_10"] = groups["PTS"].transform(previous_average, window=10)
    games["points_season_avg"] = groups["PTS"].transform(previous_season_average)
    games["minutes_last_5"] = groups["MIN"].transform(previous_average, window=5)
    games["minutes_last_10"] = groups["MIN"].transform(previous_average, window=10)
    games["fga_last_10"] = groups["FGA"].transform(previous_average, window=10)
    games["fg3a_last_10"] = groups["FG3A"].transform(previous_average, window=10)
    games["fta_last_10"] = groups["FTA"].transform(previous_average, window=10)
    games["games_played_before"] = groups.cumcount()

    games = games.merge(
        birthdates[["PLAYER_ID", "BIRTHDATE"]],
        on="PLAYER_ID", how="left", validate="many_to_one",
    )
    birthdays = pd.to_datetime(games["BIRTHDATE"])
    if birthdays.isna().any() or (birthdays >= games["GAME_DATE"]).any():
        raise ValueError("Missing or invalid player birthdate")
    # Continuous age in years; do not round to an age category.
    games["age"] = (games["GAME_DATE"] - birthdays).dt.days / 365.2425
    return games


def add_team_schedule(team_games):
    """Calculate home/away and rest using all team games, including absences."""
    teams = normalize_games(team_games)
    keys = ["season", "GAME_ID"]
    if teams.duplicated(keys + ["TEAM_ID"]).any():
        raise ValueError("Duplicate team-game rows")
    if not teams.groupby(keys).size().eq(2).all():
        raise ValueError("Each game must contain exactly two team rows")
    if not teams.groupby(keys)["GAME_DATE"].nunique().eq(1).all():
        raise ValueError("Both teams must have the same game date")

    teams = teams.sort_values(["TEAM_ID", "season", "GAME_DATE", "GAME_ID"])
    gap = teams.groupby(["TEAM_ID", "season"])["GAME_DATE"].diff().dt.days
    if (gap.dropna() < 1).any():
        raise ValueError("Team games must have distinct, increasing dates")
    teams["expected_prior_games"] = teams.groupby(["TEAM_ID", "season"]).cumcount()
    # Rest = full calendar days between games: Nov 1 -> Nov 2 means 0.
    teams["team_rest_days"] = gap - 1
    teams["team_back_to_back"] = gap.eq(1).astype(float).where(gap.notna())
    teams["is_home"] = teams["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    return teams


def add_team_ratings(teams, snapshots):
    """Attach ratings through yesterday, checking the historical cutoff."""
    snapshots = snapshots.copy()
    for column in ["GAME_DATE", "STATS_THROUGH"]:
        snapshots[column] = pd.to_datetime(snapshots[column]).dt.normalize()
    if not (snapshots["STATS_THROUGH"] < snapshots["GAME_DATE"]).all():
        raise ValueError("Team snapshots must end before the game date")

    # Exact date join prevents a later season summary being used accidentally.
    teams = teams.merge(
        snapshots[["season", "GAME_DATE", "TEAM_ID", "GP", "PACE", "DEF_RATING"]],
        on=["season", "GAME_DATE", "TEAM_ID"],
        how="left", validate="many_to_one",
    )
    # A DateTo filter that is ignored must fail instead of leaking full-season data.
    observed = teams["GP"].fillna(0)
    if not observed.eq(teams["expected_prior_games"]).all():
        raise ValueError("Snapshot games-played counts disagree with prior team games")
    has_history = teams["expected_prior_games"] > 0
    if teams.loc[has_history, ["PACE", "DEF_RATING"]].isna().any().any():
        raise ValueError("Missing advanced statistics for a team with prior games")
    teams.loc[~has_history, ["PACE", "DEF_RATING"]] = float("nan")
    return teams


def pair_opponents(teams):
    """Give each team the rest days and ratings of its opponent."""
    keys = ["season", "GAME_ID"]
    columns = keys + [
        "TEAM_ID", "GAME_DATE", "is_home", "team_rest_days",
        "team_back_to_back", "PACE", "DEF_RATING",
    ]
    teams = teams[columns]
    opponent = teams[keys + [
        "TEAM_ID", "team_rest_days", "team_back_to_back", "PACE", "DEF_RATING",
    ]].rename(columns={
        "TEAM_ID": "OPPONENT_ID",
        "team_rest_days": "opponent_rest_days",
        "team_back_to_back": "opponent_back_to_back",
        "PACE": "opponent_pace",
        "DEF_RATING": "opponent_defensive_rating",
    })
    # There are two teams per game: match each team to the other one.
    context = teams.merge(opponent, on=keys)
    context = context[context["TEAM_ID"] != context["OPPONENT_ID"]]
    return context.rename(columns={"PACE": "team_pace"})


def build_team_context(team_games, snapshots):
    teams = add_team_schedule(team_games)
    teams = add_team_ratings(teams, snapshots)
    return pair_opponents(teams)


def build_features(player_games, team_games, birthdates, snapshots):
    """Pure transformation, reusable by notebooks and batch jobs."""
    players = add_player_history(player_games, birthdates)
    context = build_team_context(team_games, snapshots)
    result = players.merge(
        context, on=["season", "GAME_ID", "TEAM_ID", "GAME_DATE"],
        how="left", validate="many_to_one", indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("Some player games have no matching team game")
    metadata = [
        "season", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ID", "TEAM_ABBREVIATION", "OPPONENT_ID",
    ]
    # Keep the outcome explicitly separate from the model-input allowlist.
    result = result.rename(columns={"PTS": "target_points"})
    return result[metadata + FEATURE_COLUMNS + ["target_points"]].rename(
        columns={column: column.lower() for column in metadata}
    )

