"""Bundled snapshot of real NBA data, used only when the live API is unreachable.

stats.nba.com blocks many cloud-host IPs (confirmed for Streamlit Community
Cloud), which would otherwise dead-end the hosted demo. The snapshot covers
exactly two players for one completed season, captured by
`python -m scripts.capture_sample_data`. Every loader returns None for any other
selection, so the app never passes off made-up data for an arbitrary pick.

Files hold each live fetcher's *post-processed* output, so a fallback returns
exactly what the live path would have. Parquet keeps dtypes exact across the
round trip (GAME_DATE stays datetime64, GAME_ID keeps its leading zeros).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
SAMPLE_SEASON = "2025-26"
SAMPLE_PLAYERS = {2544: "LeBron James", 201939: "Stephen Curry"}
SAMPLE_TEAM_IDS = {1610612747: "LAL", 1610612744: "GSW"}


def game_log_path(player_id: int, season: str) -> Path:
    return SAMPLE_DIR / f"game_log_{player_id}_{season}.parquet"


def shot_chart_path(player_id: int, season: str) -> Path:
    return SAMPLE_DIR / f"shot_chart_{player_id}_{season}.parquet"


def advanced_stats_path(season: str) -> Path:
    return SAMPLE_DIR / f"advanced_stats_{season}.parquet"


def team_advanced_stats_path(season: str) -> Path:
    return SAMPLE_DIR / f"team_advanced_stats_{season}.parquet"


def team_game_dates_path(team_id: int, season: str) -> Path:
    return SAMPLE_DIR / f"team_game_dates_{team_id}_{season}.parquet"


def _read_if_bundled(path: Path, bundled: bool) -> pd.DataFrame | None:
    if not bundled or not path.exists():
        return None
    return pd.read_parquet(path)


def load_game_log(player_id: int, season: str) -> pd.DataFrame | None:
    bundled = season == SAMPLE_SEASON and player_id in SAMPLE_PLAYERS
    return _read_if_bundled(game_log_path(player_id, season), bundled)


def load_shot_chart(player_id: int, season: str) -> pd.DataFrame | None:
    bundled = season == SAMPLE_SEASON and player_id in SAMPLE_PLAYERS
    return _read_if_bundled(shot_chart_path(player_id, season), bundled)


def load_advanced_stats(season: str) -> pd.DataFrame | None:
    return _read_if_bundled(advanced_stats_path(season), season == SAMPLE_SEASON)


def load_team_advanced_stats(season: str) -> pd.DataFrame | None:
    return _read_if_bundled(team_advanced_stats_path(season), season == SAMPLE_SEASON)


def load_team_game_dates(team_id: int, season: str) -> list | None:
    bundled = season == SAMPLE_SEASON and team_id in SAMPLE_TEAM_IDS
    df = _read_if_bundled(team_game_dates_path(team_id, season), bundled)
    return None if df is None else sorted(df["GAME_DATE"].tolist())
