"""Network I/O against nba_api, with Streamlit caching.

Every nba_api call in the app goes through this module. Each fetch has an
`_..._uncached` implementation (plain function, easy to monkeypatch in tests)
wrapped by a thin `@st.cache_data`-decorated public function.
"""
from __future__ import annotations

import random
import time

import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelog,
    shotchartdetail,
    teamgamelog,
)
from nba_api.stats.static import players, teams

import metrics

CACHE_TTL_SECONDS = 3600

# stats.nba.com is known to intermittently throttle/timeout requests from
# cloud-hosted IPs (Streamlit Community Cloud, Heroku, etc.) while working fine
# from a residential connection -- most failures clear within a few seconds
# rather than being a hard, permanent block, so a bumped timeout plus a short
# retry loop is worth it before giving up and surfacing PlayerStatsFetchError.
REQUEST_TIMEOUT_SECONDS = 45  # nba_api's own default is 30
MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SECONDS = 2

# Sliced down from nba_api's full game-log response to just what the app uses:
# GAME_DATE/PTS/REB/AST/MIN for stats and charts, MATCHUP for the Insights tab
# (opponent parsing, and the player's own team abbreviation -- PlayerGameLog has
# no TEAM_ID column at all, confirmed live, so MATCHUP is the only source for it).
GAME_LOG_COLUMNS = ["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]

# GAME_ID/PERIOD/MINUTES_REMAINING/SECONDS_REMAINING/SHOT_MADE_FLAG for the hot-hand
# permutation test, LOC_X/LOC_Y for the shot chart heatmap -- both insights share
# this one fetch, so selecting a player on the Insights page costs one network call.
SHOT_CHART_COLUMNS = [
    "GAME_ID",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "SHOT_MADE_FLAG",
    "LOC_X",
    "LOC_Y",
]


class PlayerStatsFetchError(Exception):
    """Raised when an nba_api call fails (rate limit, timeout, malformed response)."""


def _call_with_retries(fetch_fn):
    """Retry a flaky nba_api call with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fetch_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 1))
    raise last_exc


def _get_all_players_uncached() -> list[dict]:
    try:
        return players.get_players()
    except Exception as exc:
        raise PlayerStatsFetchError(f"Could not fetch player list: {exc}") from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_all_players() -> list[dict]:
    return _get_all_players_uncached()


def _fetch_player_game_log_uncached(player_id: int, season: str):
    try:
        df = _call_with_retries(
            lambda: playergamelog.PlayerGameLog(
                player_id=player_id, season=season, timeout=REQUEST_TIMEOUT_SECONDS
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(
            f"Could not fetch game log for player {player_id}, season {season}: {exc}"
        ) from exc
    df = df[[c for c in GAME_LOG_COLUMNS if c in df.columns]]
    return metrics.parse_and_sort_game_log(df)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_player_game_log(player_id: int, season: str):
    return _fetch_player_game_log_uncached(player_id, season)


def _fetch_advanced_stats_uncached(season: str):
    try:
        return _call_with_retries(
            lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(
            f"Could not fetch advanced stats for season {season}: {exc}"
        ) from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_advanced_stats(season: str):
    """Whole-league advanced stats for a season, fetched once and shared across players."""
    return _fetch_advanced_stats_uncached(season)


def _get_all_teams_uncached() -> list[dict]:
    try:
        return teams.get_teams()
    except Exception as exc:
        raise PlayerStatsFetchError(f"Could not fetch team list: {exc}") from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_all_teams() -> list[dict]:
    return _get_all_teams_uncached()


def _fetch_team_game_dates_uncached(team_id: int, season: str) -> list:
    try:
        df = _call_with_retries(
            lambda: teamgamelog.TeamGameLog(
                team_id=team_id, season=season, timeout=REQUEST_TIMEOUT_SECONDS
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(
            f"Could not fetch team schedule for team {team_id}, season {season}: {exc}"
        ) from exc
    dates = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    return sorted(dates.tolist())


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_team_game_dates(team_id: int, season: str) -> list:
    """Every date the given team played in `season` (Insights: Absence & Rest Impact)."""
    return _fetch_team_game_dates_uncached(team_id, season)


def _fetch_team_advanced_stats_uncached(season: str):
    try:
        return _call_with_retries(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(
            f"Could not fetch team advanced stats for season {season}: {exc}"
        ) from exc


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_team_advanced_stats(season: str):
    """Whole-league team DEF_RATING for a season (Insights: Performance vs. Defensive Quality)."""
    return _fetch_team_advanced_stats_uncached(season)


def _fetch_shot_chart_uncached(player_id: int, season: str):
    try:
        df = _call_with_retries(
            lambda: shotchartdetail.ShotChartDetail(
                team_id=0,  # 0 = all teams, sidesteps mid-season trades for this analysis
                player_id=player_id,
                season_nullable=season,
                context_measure_simple="FGA",
                timeout=REQUEST_TIMEOUT_SECONDS,
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(
            f"Could not fetch shot chart for player {player_id}, season {season}: {exc}"
        ) from exc
    return df[[c for c in SHOT_CHART_COLUMNS if c in df.columns]]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_shot_chart(player_id: int, season: str):
    """Every shot attempt for a player-season (Insights: Hot Hand test + Shot Chart)."""
    return _fetch_shot_chart_uncached(player_id, season)
