import pandas as pd
import pytest

import data


class _FakeGameLog:
    def __init__(self, player_id, season):
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
    def __init__(self, season, measure_type_detailed_defense):
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
    def __init__(self, team_id, season):
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
    def __init__(self, season, measure_type_detailed_defense):
        self.season = season
        self.measure_type_detailed_defense = measure_type_detailed_defense

    def get_data_frames(self):
        df = pd.DataFrame({"TEAM_ID": [1610612747], "DEF_RATING": [112.0]})
        return [df]


class _FakeShotChart:
    def __init__(self, team_id, player_id, season_nullable, context_measure_simple):
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
