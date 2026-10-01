"""Network I/O against nba_api, with Streamlit caching.

Every nba_api call in the app goes through this module. Each fetch has an
`_..._uncached` implementation (plain function, easy to monkeypatch in tests)
wrapped by a thin `@st.cache_data`-decorated public function.
"""
from __future__ import annotations

import random
import time

import pandas as pd
import requests
import streamlit as st
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelog,
    playergamelogs,
    shotchartdetail,
    teamgamelog,
)
from nba_api.stats.static import players, teams

import metrics
import sample_data_loader

CACHE_TTL_SECONDS = 3600

# stats.nba.com hard-blocks many cloud-host IPs (confirmed on Streamlit
# Community Cloud: requests hang until they time out, every time), while a
# residential connection answers in a second or two. So: a modest timeout, one
# retry for genuine blips, then a circuit breaker so a blocked host fails fast
# on every later call instead of waiting out the timeout again per fetch.
REQUEST_TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 2
RETRY_BACKOFF_BASE_SECONDS = 2
CIRCUIT_COOLDOWN_SECONDS = 600

# Offline-only (scripts/train_model.py): the league-wide pull is ~30k rows.
LEAGUE_REQUEST_TIMEOUT_SECONDS = 90

_api_unreachable_until = 0.0

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


class CircuitOpenError(Exception):
    """Raised without touching the network while the circuit breaker is open."""


def _call_with_retries(fetch_fn):
    """Retry a flaky nba_api call with exponential backoff + jitter.

    Only network-level failures (timeouts, refused connections) open the
    circuit -- a malformed response for one player shouldn't black out the API
    for everyone else.
    """
    global _api_unreachable_until
    if time.monotonic() < _api_unreachable_until:
        raise CircuitOpenError(
            "stats.nba.com stopped responding moments ago; skipping live calls for a few minutes"
        )
    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fetch_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0, 1))
    if isinstance(last_exc, requests.exceptions.RequestException):
        _api_unreachable_until = time.monotonic() + CIRCUIT_COOLDOWN_SECONDS
    raise last_exc


def _with_sample_fallback(live_fn, sample_fn, *args):
    """(result, used_sample_data). Live data first; on failure, the bundled
    snapshot if this exact selection is in it, otherwise re-raise.

    Deliberately runs outside the cache: only successful live fetches are cached
    (via _live_cached), so the app returns to live data as soon as the API does.
    While it's down, the open circuit makes each retry instant and the snapshot is
    a few-millisecond Parquet read.
    """
    try:
        return live_fn(*args), False
    except PlayerStatsFetchError:
        sample = sample_fn(*args)
        if sample is None:
            raise
        return sample, True


_live_cached = st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching live NBA data...")


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


_cached_player_game_log = _live_cached(_fetch_player_game_log_uncached)


def fetch_player_game_log(player_id: int, season: str):
    """(game log, used_sample_data)."""
    return _with_sample_fallback(_cached_player_game_log, sample_data_loader.load_game_log, player_id, season)


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


_cached_advanced_stats = _live_cached(_fetch_advanced_stats_uncached)


def fetch_advanced_stats(season: str):
    """(whole-league advanced stats, used_sample_data) -- fetched once per season, shared across players."""
    return _with_sample_fallback(_cached_advanced_stats, sample_data_loader.load_advanced_stats, season)


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


_cached_team_game_dates = _live_cached(_fetch_team_game_dates_uncached)


def fetch_team_game_dates(team_id: int, season: str):
    """(every date the team played in `season`, used_sample_data) -- Insights: Absence & Rest Impact."""
    return _with_sample_fallback(
        _cached_team_game_dates, sample_data_loader.load_team_game_dates, team_id, season
    )


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


_cached_team_advanced_stats = _live_cached(_fetch_team_advanced_stats_uncached)


def fetch_team_advanced_stats(season: str):
    """(whole-league team DEF_RATING, used_sample_data) -- Insights: Performance vs. Defensive Quality."""
    return _with_sample_fallback(
        _cached_team_advanced_stats, sample_data_loader.load_team_advanced_stats, season
    )


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


_cached_shot_chart = _live_cached(_fetch_shot_chart_uncached)


def fetch_shot_chart(player_id: int, season: str):
    """(every shot attempt for a player-season, used_sample_data) -- Insights: Hot Hand + Shot Chart."""
    return _with_sample_fallback(_cached_shot_chart, sample_data_loader.load_shot_chart, player_id, season)


LEAGUE_GAME_LOG_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "MIN",
    "PTS",
    "REB",
    "AST",
]


def fetch_league_game_logs(season: str, timeout: int = LEAGUE_REQUEST_TIMEOUT_SECONDS) -> pd.DataFrame:
    """Every player's regular-season game log for `season`, in one call (model training data).

    Offline-only -- scripts/train_model.py is the sole caller, so no st.cache_data.
    The plural endpoint returns ISO dates (unlike the singular endpoints' "OCT 24, 2023")
    and includes playoff games unless told otherwise; the live app only ever sees
    regular-season logs, so the model is trained on the same.
    """
    try:
        df = _call_with_retries(
            lambda: playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=timeout,
            ).get_data_frames()[0]
        )
    except Exception as exc:
        raise PlayerStatsFetchError(f"Could not fetch league game logs for season {season}: {exc}") from exc
    df = df[[c for c in LEAGUE_GAME_LOG_COLUMNS if c in df.columns]]
    return metrics.parse_and_sort_game_log(df, date_format=None)
