import pandas as pd

import data
import sample_data_loader as sdl
from insights import defense_tiers

SEASON = sdl.SAMPLE_SEASON


def test_bundle_is_exactly_two_players_one_season():
    assert sdl.SAMPLE_PLAYERS == {2544: "LeBron James", 201939: "Stephen Curry"}
    assert SEASON == "2025-26"


def test_game_log_loads_with_live_shape_for_bundled_players():
    for player_id in sdl.SAMPLE_PLAYERS:
        df = sdl.load_game_log(player_id, SEASON)
        assert list(df.columns) == data.GAME_LOG_COLUMNS
        assert pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"])
        assert df["GAME_DATE"].is_monotonic_increasing
        assert len(df) > 20


def test_bundled_players_stayed_on_one_team_all_season():
    # The Absence insight assumes one team all season; the bundle must honor that.
    for player_id in sdl.SAMPLE_PLAYERS:
        df = sdl.load_game_log(player_id, SEASON)
        assert df["MATCHUP"].map(defense_tiers.extract_own_team_abbreviation).nunique() == 1


def test_shot_chart_keeps_game_id_leading_zeros():
    df = sdl.load_shot_chart(2544, SEASON)
    assert list(df.columns) == data.SHOT_CHART_COLUMNS
    assert df["GAME_ID"].str.startswith("00").all()


def test_league_tables_are_complete_not_sliced():
    assert sdl.load_team_advanced_stats(SEASON)["TEAM_ID"].nunique() == 30
    advanced = sdl.load_advanced_stats(SEASON)
    assert set(sdl.SAMPLE_PLAYERS) <= set(advanced["PLAYER_ID"])
    assert len(advanced) > 400


def test_team_game_dates_are_sorted_timestamps():
    for team_id in sdl.SAMPLE_TEAM_IDS:
        dates = sdl.load_team_game_dates(team_id, SEASON)
        assert dates == sorted(dates)
        assert all(isinstance(d, pd.Timestamp) for d in dates)


def test_non_bundled_selections_return_none():
    assert sdl.load_game_log(203999, SEASON) is None  # a real player, just not bundled
    assert sdl.load_game_log(2544, "2024-25") is None
    assert sdl.load_shot_chart(201939, "2023-24") is None
    assert sdl.load_advanced_stats("2024-25") is None
    assert sdl.load_team_advanced_stats("2024-25") is None
    assert sdl.load_team_game_dates(1610612738, SEASON) is None
