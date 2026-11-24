"""Small invented NBA-style records; these are test fixtures, not research data."""

import sys
import pandas as pd

def fixture_tables(season):
    """Four invented players and two invented teams in twelve invented games."""
    players, teams, snapshots = [], [], []
    year = int(season[:4])
    for i in range(12):
        date = pd.Timestamp(f"{year}-11-01") + pd.Timedelta(days=2 * i)
        game_id = f"002{year % 100:02d}{i + 1:05d}"
        for team in [100, 200]:
            common = {"season": season, "GAME_ID": game_id, "GAME_DATE": date, "TEAM_ID": team}
            teams.append({**common, "MATCHUP": "AAA vs. BBB" if team == 100 else "BBB @ AAA"})
            snapshots.append({"season": season, "GAME_DATE": date,
                "STATS_THROUGH": date - pd.Timedelta(days=1), "TEAM_ID": team,
                "GP": i, "PACE": 98 + i / 10, "DEF_RATING": 108 + team / 100 + i / 5})
            for offset in [1, 2]:
                players.append({**common, "PLAYER_ID": team + offset,
                    "PLAYER_NAME": f"Fixture Player {team + offset}",
                    "TEAM_ABBREVIATION": "AAA" if team == 100 else "BBB",
                    "PTS": 5 + i + offset, "MIN": 15 + i + offset,
                    "FGA": 6 + i, "FG3A": 2 + i % 3, "FTA": 1 + offset})
    birthdates = pd.DataFrame({"PLAYER_ID": [101, 102, 201, 202],
                              "BIRTHDATE": ["1990-01-01"] * 4})
    return pd.DataFrame(players), pd.DataFrame(teams), birthdates, pd.DataFrame(snapshots)


class FixtureSource:
    def games(self, season, kind, refresh=False):
        return fixture_tables(season)[0 if kind == "P" else 1]

    def birthdates(self, player_ids):
        return fixture_tables("2020-21")[2]

    def team_snapshots(self, season, game_dates):
        return fixture_tables(season)[3]

def execute_notebook(notebook, cwd, timeout=180):
    """Use the test process's Python environment, including on Windows."""
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        return NotebookClient(notebook, km=manager, timeout=timeout,
            resources={"metadata": {"path": str(cwd)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
