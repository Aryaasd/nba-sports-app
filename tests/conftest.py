import pandas as pd
import pytest


@pytest.fixture
def fake_players():
    return [
        {"id": 2544, "full_name": "LeBron James"},
        {"id": 201142, "full_name": "Kevin Durant"},
        {"id": 201939, "full_name": "Stephen Curry"},
    ]


@pytest.fixture
def fake_game_log_df():
    # Deliberately out of chronological order, string dates as nba_api returns them.
    return pd.DataFrame(
        {
            "GAME_DATE": ["APR 09, 2024", "OCT 24, 2023", "JAN 15, 2024"],
            "PTS": [30, 20, 25],
            "REB": [8, 6, 7],
            "AST": [9, 5, 6],
            "MIN": [36.0, 34.0, 35.0],
        }
    )


@pytest.fixture
def fake_teams():
    return [
        {"id": 1610612737, "abbreviation": "ATL"},
        {"id": 1610612747, "abbreviation": "LAL"},
    ]


@pytest.fixture
def fake_advanced_stats_df():
    return pd.DataFrame(
        {
            "PLAYER_ID": [2544, 201142],
            "PLAYER_NAME": ["LeBron James", "Kevin Durant"],
            "TS_PCT": [0.62, 0.64],
            "USG_PCT": [0.31, 0.29],
            "PACE": [98.5, 99.2],
            "PIE": [0.155, 0.148],
        }
    )
