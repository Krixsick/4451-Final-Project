"""NBA downloads and cache handling. Feature math lives in features.py."""

import time

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguegamelog, leaguedashteamstats
from requests.exceptions import RequestException


def save_csv(frame, destination):
    """Replace a completed export atomically; interrupted writes stay separate."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(destination)


class NBAData:
    """Cache NBA source tables so experiments do not repeat every download."""

    def __init__(self, folder, offline=False, timeout=30, attempts=2):
        self.folder = folder
        self.offline = offline
        self.timeout = timeout
        self.attempts = attempts

    def table(self, relative_path, fetch, required, refresh=False):
        path = self.folder / relative_path
        if path.exists() and not refresh:
            frame = pd.read_csv(path, dtype={"GAME_ID": "string"})
        else:
            if self.offline:
                raise FileNotFoundError(f"Offline mode: source cache missing: {path}")
            for attempt in range(1, self.attempts + 1):
                print(f"Fetching {relative_path} (attempt {attempt})", flush=True)
                try:
                    frame = fetch()
                    self.validate(frame, required, path)
                    save_csv(frame, path)
                    time.sleep(1)
                    break
                except (RequestException, ValueError, KeyError) as error:
                    if attempt == self.attempts:
                        raise RuntimeError(f"NBA download failed: {relative_path}: {error}") from error
                    time.sleep(2 * attempt)
        self.validate(frame, required, path)
        return frame

    @staticmethod
    def validate(frame, required, path):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing source columns {sorted(missing)}")

    def games(self, season, kind, refresh=False):
        required = ["GAME_ID", "GAME_DATE", "TEAM_ID"]
        if kind == "P":
            required += ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "MIN", "FGA", "FG3A", "FTA"]
        else:
            required += ["MATCHUP"]
        label = "players" if kind == "P" else "teams"
        frame = self.table(
            f"{season}/{label}.csv",
            lambda: leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation=kind, timeout=self.timeout,
            ).get_data_frames()[0],
            required, refresh,
        )
        if frame.empty:
            raise ValueError(f"No regular-season {label} returned for {season}")
        return frame.assign(season=season)

    def birthdates(self, player_ids):
        profiles = []
        for player_id in sorted(set(player_ids)):
            profile = self.table(
                f"profiles/{player_id}.csv",
                lambda: commonplayerinfo.CommonPlayerInfo(
                    player_id=int(player_id), timeout=self.timeout,
                ).common_player_info.get_data_frame(),
                ["PERSON_ID", "BIRTHDATE"],
            )
            if len(profile) != 1 or int(profile.iloc[0]["PERSON_ID"]) != int(player_id):
                raise ValueError(f"Unexpected profile returned for player {player_id}")
            profiles.append(profile[["PERSON_ID", "BIRTHDATE"]])
        return pd.concat(profiles, ignore_index=True).rename(columns={"PERSON_ID": "PLAYER_ID"})

    def team_snapshots(self, season, game_dates):
        snapshots = []
        for game_date in sorted(pd.DatetimeIndex(game_dates).normalize().unique()):
            game_date = pd.Timestamp(game_date)
            cutoff = game_date - pd.Timedelta(days=1)
            frame = self.table(
                f"{season}/team_stats/{cutoff:%Y-%m-%d}.csv",
                lambda: leaguedashteamstats.LeagueDashTeamStats(
                    season=season, season_type_all_star="Regular Season",
                    measure_type_detailed_defense="Advanced",
                    per_mode_detailed="PerGame",
                    date_to_nullable=cutoff.strftime("%m/%d/%Y"),
                    timeout=self.timeout,
                ).get_data_frames()[0],
                ["TEAM_ID", "GP", "DEF_RATING", "PACE"],
            )
            frame = frame[["TEAM_ID", "GP", "DEF_RATING", "PACE"]].copy()
            frame["season"] = season
            frame["GAME_DATE"] = game_date
            frame["STATS_THROUGH"] = cutoff
            snapshots.append(frame)
        return pd.concat(snapshots, ignore_index=True)

