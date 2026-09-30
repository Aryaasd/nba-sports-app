import time

import pandas as pd
import pytest
import requests

import data


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Retries sleep with backoff between attempts -- skip the real delay so
    # the exception-wrapping tests (which exhaust all retries) stay fast.
    monkeypatch.setattr(data.time, "sleep", lambda _: None)


@pytest.fixture(autouse=True)
def _closed_circuit(monkeypatch):
    monkeypatch.setattr(data, "_api_unreachable_until", 0.0)


class _FakeGameLog:
    def __init__(self, player_id, season, timeout=None):
        self.player_id = player_id
        self.season = season

    def get_data_frames(self):
        df = pd.DataFrame(
            {
                "GAME_DATE": ["APR 09, 2024", "OCT 24, 2023"],
                "MATCHUP": ["LAL vs. HOU", "LAL @ DAL"],
                "PTS": [30, 20],
                "REB": [8, 6],
                "AST": [9, 5],
                "MIN": [36.0, 34.0],
            }
        )
        return [df]


class _FakeAdvancedStats:
    def __init__(self, season, measure_type_detailed_defense, timeout=None):
        self.season = season
        self.measure_type_detailed_defense = measure_type_detailed_defense

    def get_data_frames(self):
        df = pd.DataFrame(
            {
                "PLAYER_ID": [2544],
                "TS_PCT": [0.62],
                "USG_PCT": [0.31],
                "PACE": [98.5],
                "PIE": [0.155],
            }
        )
        return [df]


class _FakeTeamGameLog:
    def __init__(self, team_id, season, timeout=None):
        self.team_id = team_id
        self.season = season

    def get_data_frames(self):
        df = pd.DataFrame(
            {
                "GAME_DATE": ["APR 14, 2024", "OCT 24, 2023"],
                "MATCHUP": ["LAL @ NOP", "LAL vs. GSW"],
            }
        )
        return [df]


class _FakeTeamAdvancedStats:
    def __init__(self, season, measure_type_detailed_defense, timeout=None):
        self.season = season
        self.measure_type_detailed_defense = measure_type_detailed_defense

    def get_data_frames(self):
        df = pd.DataFrame({"TEAM_ID": [1610612747], "DEF_RATING": [112.0]})
        return [df]


class _FakeShotChart:
    def __init__(self, team_id, player_id, season_nullable, context_measure_simple, timeout=None):
        self.team_id = team_id
        self.player_id = player_id
        self.season_nullable = season_nullable
        self.context_measure_simple = context_measure_simple

    def get_data_frames(self):
        df = pd.DataFrame(
            {
                "GAME_ID": ["0022300001", "0022300001"],
                "PERIOD": [1, 1],
                "MINUTES_REMAINING": [11, 10],
                "SECONDS_REMAINING": [30, 15],
                "SHOT_MADE_FLAG": [1, 0],
                "LOC_X": [10, -20],
                "LOC_Y": [50, 80],
            }
        )
        return [df]


class _RaisesOnInit:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("simulated nba_api failure")


# --- _call_with_retries -------------------------------------------------------

def test_call_with_retries_succeeds_after_transient_failures():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < data.MAX_ATTEMPTS:
            raise TimeoutError("simulated transient timeout")
        return "ok"

    assert data._call_with_retries(flaky) == "ok"
    assert calls["count"] == data.MAX_ATTEMPTS


def test_call_with_retries_raises_after_exhausting_attempts():
    calls = {"count": 0}

    def always_fails():
        calls["count"] += 1
        raise TimeoutError("simulated permanent timeout")

    with pytest.raises(TimeoutError):
        data._call_with_retries(always_fails)
    assert calls["count"] == data.MAX_ATTEMPTS


# --- circuit breaker ------------------------------------------------------------

def test_circuit_opens_after_network_failures_and_next_call_skips_the_network():
    calls = {"count": 0}

    def blocked_host():
        calls["count"] += 1
        raise requests.exceptions.ReadTimeout("simulated cloud-IP block")

    with pytest.raises(requests.exceptions.ReadTimeout):
        data._call_with_retries(blocked_host)
    assert calls["count"] == data.MAX_ATTEMPTS
    assert data._api_unreachable_until > time.monotonic() + data.CIRCUIT_COOLDOWN_SECONDS - 60

    with pytest.raises(data.CircuitOpenError):
        data._call_with_retries(blocked_host)
    assert calls["count"] == data.MAX_ATTEMPTS  # fail-fast: never reached the network again


def test_non_network_error_does_not_open_circuit():
    def malformed_response():
        raise KeyError("resultSets")

    with pytest.raises(KeyError):
        data._call_with_retries(malformed_response)
    assert data._call_with_retries(lambda: "ok") == "ok"


def test_circuit_closes_after_cooldown(monkeypatch):
    monkeypatch.setattr(data, "_api_unreachable_until", time.monotonic() - 1)
    assert data._call_with_retries(lambda: "ok") == "ok"


def test_open_circuit_surfaces_as_fetch_error(monkeypatch):
    monkeypatch.setattr(data, "_api_unreachable_until", time.monotonic() + 600)
    with pytest.raises(data.PlayerStatsFetchError, match="skipping live calls"):
        data._fetch_player_game_log_uncached(2544, "2023-24")


# --- sample-data fallback --------------------------------------------------------

def _live_fails(*args):
    raise data.PlayerStatsFetchError("simulated live failure")


def test_fallback_not_used_when_live_fetch_succeeds():
    sample_calls = []
    result = data._with_sample_fallback(lambda *a: "live", lambda *a: sample_calls.append(a), 2544, "2025-26")
    assert result == ("live", False)
    assert sample_calls == []


def test_fallback_returns_sample_for_bundled_selection():
    assert data._with_sample_fallback(_live_fails, lambda *a: "sample", 2544, "2025-26") == ("sample", True)


def test_fallback_still_raises_for_non_bundled_selection():
    # The core promise: an arbitrary selection is never answered with made-up data.
    with pytest.raises(data.PlayerStatsFetchError, match="simulated live failure"):
        data._with_sample_fallback(_live_fails, lambda *a: None, 203999, "2025-26")


@pytest.fixture
def _fresh_cache():
    data.fetch_player_game_log.clear()
    yield
    data.fetch_player_game_log.clear()


def test_public_fetcher_falls_back_to_bundled_snapshot(monkeypatch, _fresh_cache):
    monkeypatch.setattr(data.playergamelog, "PlayerGameLog", _RaisesOnInit)
    df, used_sample = data.fetch_player_game_log(2544, "2025-26")
    assert used_sample is True
    assert list(df.columns) == data.GAME_LOG_COLUMNS
    assert len(df) > 20


def test_public_fetcher_raises_for_non_bundled_selection(monkeypatch, _fresh_cache):
    monkeypatch.setattr(data.playergamelog, "PlayerGameLog", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data.fetch_player_game_log(2544, "2023-24")


def test_public_fetcher_live_success_flags_no_sample(monkeypatch, _fresh_cache):
    monkeypatch.setattr(data.playergamelog, "PlayerGameLog", _FakeGameLog)
    df, used_sample = data.fetch_player_game_log(2544, "2025-26")
    assert used_sample is False
    assert len(df) == 2


# --- get_all_players --------------------------------------------------------

def test_get_all_players_uncached_passthrough(monkeypatch):
    fake = [{"id": 1, "full_name": "Test Player"}]
    monkeypatch.setattr(data.players, "get_players", lambda: fake)
    assert data._get_all_players_uncached() == fake


def test_get_all_players_uncached_wraps_exception(monkeypatch):
    def boom():
        raise RuntimeError("network down")

    monkeypatch.setattr(data.players, "get_players", boom)
    with pytest.raises(data.PlayerStatsFetchError):
        data._get_all_players_uncached()


# --- fetch_player_game_log ---------------------------------------------------

def test_fetch_player_game_log_uncached_parsed_and_sorted(monkeypatch):
    monkeypatch.setattr(data.playergamelog, "PlayerGameLog", _FakeGameLog)
    result = data._fetch_player_game_log_uncached(2544, "2023-24")
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])
    assert list(result["GAME_DATE"]) == sorted(result["GAME_DATE"])
    assert list(result.columns) == data.GAME_LOG_COLUMNS
    assert result.iloc[0]["PTS"] == 20  # Oct game is earliest, sorted to the front


def test_fetch_player_game_log_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.playergamelog, "PlayerGameLog", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._fetch_player_game_log_uncached(2544, "2023-24")


# --- fetch_advanced_stats -----------------------------------------------------

def test_fetch_advanced_stats_uncached_passthrough(monkeypatch):
    monkeypatch.setattr(data.leaguedashplayerstats, "LeagueDashPlayerStats", _FakeAdvancedStats)
    result = data._fetch_advanced_stats_uncached("2023-24")
    assert result.loc[0, "TS_PCT"] == pytest.approx(0.62)


def test_fetch_advanced_stats_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.leaguedashplayerstats, "LeagueDashPlayerStats", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._fetch_advanced_stats_uncached("2023-24")


# --- get_all_teams -----------------------------------------------------------

def test_get_all_teams_uncached_passthrough(monkeypatch):
    fake = [{"id": 1610612747, "abbreviation": "LAL"}]
    monkeypatch.setattr(data.teams, "get_teams", lambda: fake)
    assert data._get_all_teams_uncached() == fake


def test_get_all_teams_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.teams, "get_teams", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._get_all_teams_uncached()


# --- fetch_team_game_dates -----------------------------------------------------

def test_fetch_team_game_dates_uncached_parsed_and_sorted(monkeypatch):
    monkeypatch.setattr(data.teamgamelog, "TeamGameLog", _FakeTeamGameLog)
    result = data._fetch_team_game_dates_uncached(1610612747, "2023-24")
    assert result == sorted(result)
    assert result[0] == pd.Timestamp("2023-10-24")
    assert result[-1] == pd.Timestamp("2024-04-14")


def test_fetch_team_game_dates_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.teamgamelog, "TeamGameLog", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._fetch_team_game_dates_uncached(1610612747, "2023-24")


# --- fetch_team_advanced_stats --------------------------------------------------

def test_fetch_team_advanced_stats_uncached_passthrough(monkeypatch):
    monkeypatch.setattr(data.leaguedashteamstats, "LeagueDashTeamStats", _FakeTeamAdvancedStats)
    result = data._fetch_team_advanced_stats_uncached("2023-24")
    assert result.loc[0, "DEF_RATING"] == pytest.approx(112.0)


def test_fetch_team_advanced_stats_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.leaguedashteamstats, "LeagueDashTeamStats", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._fetch_team_advanced_stats_uncached("2023-24")


# --- fetch_shot_chart ----------------------------------------------------------

def test_fetch_shot_chart_uncached_passthrough(monkeypatch):
    monkeypatch.setattr(data.shotchartdetail, "ShotChartDetail", _FakeShotChart)
    result = data._fetch_shot_chart_uncached(2544, "2023-24")
    assert list(result.columns) == data.SHOT_CHART_COLUMNS
    assert len(result) == 2
    assert result.iloc[0]["SHOT_MADE_FLAG"] == 1


def test_fetch_shot_chart_uncached_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.shotchartdetail, "ShotChartDetail", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data._fetch_shot_chart_uncached(2544, "2023-24")


# --- fetch_league_game_logs (model training data) ---------------------------------

class _FakeLeagueGameLogs:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeLeagueGameLogs.last_kwargs = kwargs

    def get_data_frames(self):
        df = pd.DataFrame(
            {
                "SEASON_YEAR": ["2024-25"] * 3,
                "PLAYER_ID": [1, 2, 1],
                "PLAYER_NAME": ["A", "B", "A"],
                "TEAM_ID": [10, 20, 10],
                "GAME_ID": ["0022400002", "0022400001", "0022400001"],
                "GAME_DATE": ["2024-10-24T00:00:00", "2024-10-22T00:00:00", "2024-10-22T00:00:00"],
                "MATCHUP": ["AAA @ BBB", "BBB vs. AAA", "AAA @ BBB"],
                "MIN": [30.5, 28.25, 31.0],
                "PTS": [20, 15, 18],
                "REB": [5, 7, 4],
                "AST": [3, 2, 6],
            }
        )
        return [df]


def test_fetch_league_game_logs_regular_season_iso_dates_sorted(monkeypatch):
    monkeypatch.setattr(data.playergamelogs, "PlayerGameLogs", _FakeLeagueGameLogs)
    result = data.fetch_league_game_logs("2024-25")
    assert _FakeLeagueGameLogs.last_kwargs["season_type_nullable"] == "Regular Season"
    assert list(result.columns) == data.LEAGUE_GAME_LOG_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])
    assert result["GAME_DATE"].is_monotonic_increasing


def test_fetch_league_game_logs_wraps_exception(monkeypatch):
    monkeypatch.setattr(data.playergamelogs, "PlayerGameLogs", _RaisesOnInit)
    with pytest.raises(data.PlayerStatsFetchError):
        data.fetch_league_game_logs("2024-25")
