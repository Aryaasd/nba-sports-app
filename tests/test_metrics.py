import datetime as dt

import numpy as np
import pandas as pd
import pytest

import metrics

# --- season strings ---------------------------------------------------

@pytest.mark.parametrize(
    "today, expected",
    [
        (dt.date(2026, 8, 5), "2025-26"),   # off-season, month < 10
        (dt.date(2025, 9, 30), "2024-25"),  # last day before the Oct cutoff
        (dt.date(2025, 10, 1), "2025-26"),  # first day of the new season
        (dt.date(2024, 12, 25), "2024-25"), # mid-season, month >= 10
    ],
)
def test_get_current_season(today, expected):
    assert metrics.get_current_season(today) == expected


@pytest.mark.parametrize(
    "today, expected",
    [
        (dt.date(2026, 9, 30), "2025-26"),
        (dt.date(2026, 10, 1), "2025-26"),  # new season listed, but no games yet
        (dt.date(2026, 10, 31), "2025-26"),
        (dt.date(2026, 11, 1), "2026-27"),
        (dt.date(2027, 3, 15), "2026-27"),
    ],
)
def test_get_default_season_skips_empty_october(today, expected):
    assert metrics.get_default_season(today) == expected
    assert expected in metrics.get_recent_seasons(today=today)


def test_get_recent_seasons_ordering():
    seasons = metrics.get_recent_seasons(n=5, today=dt.date(2026, 8, 5))
    assert seasons == ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22"]


def test_get_recent_seasons_length():
    assert len(metrics.get_recent_seasons(n=3, today=dt.date(2026, 8, 5))) == 3


# --- player lookup ------------------------------------------------------

def test_build_player_index_and_lookup(fake_players):
    index = metrics.build_player_index(fake_players)
    assert metrics.get_player_id(index, "LeBron James") == 2544
    assert metrics.get_player_id(index, "lebron james") == 2544  # case-insensitive


def test_get_player_id_unknown_name(fake_players):
    index = metrics.build_player_index(fake_players)
    assert metrics.get_player_id(index, "Not A Player") is None


def test_get_player_names_sorted(fake_players):
    names = metrics.get_player_names(fake_players)
    assert names == sorted(names)
    assert "LeBron James" in names


def test_build_team_abbreviation_index(fake_teams):
    index = metrics.build_team_abbreviation_index(fake_teams)
    assert index == {1610612737: "ATL", 1610612747: "LAL"}


def test_build_abbreviation_to_team_id_index(fake_teams):
    index = metrics.build_abbreviation_to_team_id_index(fake_teams)
    assert index == {"ATL": 1610612737, "LAL": 1610612747}


def test_safe_selectbox_index_present():
    assert metrics.safe_selectbox_index(["A", "B", "C"], "B") == 1


def test_safe_selectbox_index_absent_falls_back():
    assert metrics.safe_selectbox_index(["A", "B", "C"], "Z") == 0
    assert metrics.safe_selectbox_index(["A", "B", "C"], "Z", fallback=2) == 2


# --- game log parsing -----------------------------------------------------

def test_parse_and_sort_game_log_orders_ascending(fake_game_log_df):
    result = metrics.parse_and_sort_game_log(fake_game_log_df)
    assert list(result["GAME_DATE"]) == sorted(result["GAME_DATE"])
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])
    # earliest date first
    assert result.iloc[0]["GAME_DATE"] == pd.Timestamp("2023-10-24")


def test_parse_and_sort_game_log_iso_dates_with_date_format_none():
    # PlayerGameLogs (league-wide) returns ISO timestamps, not "OCT 24, 2023".
    df = pd.DataFrame(
        {"GAME_DATE": ["2025-06-22T00:00:00", "2024-10-22T00:00:00", "2025-01-15T00:00:00"], "PTS": [1, 2, 3]}
    )
    result = metrics.parse_and_sort_game_log(df, date_format=None)
    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE"])
    assert list(result["GAME_DATE"]) == [
        pd.Timestamp("2024-10-22"),
        pd.Timestamp("2025-01-15"),
        pd.Timestamp("2025-06-22"),
    ]
    assert list(result["PTS"]) == [2, 3, 1]


def test_parse_and_sort_game_log_empty_df_safe():
    empty = pd.DataFrame(columns=["GAME_DATE", "PTS"])
    result = metrics.parse_and_sort_game_log(empty)
    assert result.empty


# --- stat math ------------------------------------------------------------

def test_true_shooting_pct_known_value():
    assert metrics.true_shooting_pct(30, 20, 5) == pytest.approx(0.67567567)


def test_true_shooting_pct_zero_denominator_is_nan():
    assert np.isnan(metrics.true_shooting_pct(0, 0, 0))


def test_true_shooting_pct_vectorized_on_series():
    pts = pd.Series([30, 0])
    fga = pd.Series([20, 0])
    fta = pd.Series([5, 0])
    result = metrics.true_shooting_pct(pts, fga, fta)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == pytest.approx(0.67567567)
    assert np.isnan(result.iloc[1])


def test_per_36_known_value():
    assert metrics.per_36(20, 30) == pytest.approx(24.0)


def test_per_36_zero_minutes_is_nan():
    assert np.isnan(metrics.per_36(10, 0))


def test_per_36_vectorized_on_series():
    result = metrics.per_36(pd.Series([20, 10]), pd.Series([30, 0]))
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == pytest.approx(24.0)
    assert np.isnan(result.iloc[1])


def test_season_per36_totals(fake_game_log_df):
    totals = metrics.season_per36_totals(fake_game_log_df)
    assert totals["PTS/36"] == pytest.approx(75 / 105 * 36)
    assert totals["REB/36"] == pytest.approx(21 / 105 * 36)
    assert totals["AST/36"] == pytest.approx(20 / 105 * 36)


def test_season_per36_totals_empty_df_returns_nan():
    empty = pd.DataFrame(columns=["PTS", "REB", "AST", "MIN"])
    totals = metrics.season_per36_totals(empty)
    assert all(np.isnan(v) for v in totals.values())


# --- advanced metrics -------------------------------------------------------

def test_extract_player_advanced_row_found(fake_advanced_stats_df):
    row = metrics.extract_player_advanced_row(fake_advanced_stats_df, 2544)
    assert row is not None
    assert row["TS_PCT"] == pytest.approx(0.62)


def test_extract_player_advanced_row_not_found(fake_advanced_stats_df):
    assert metrics.extract_player_advanced_row(fake_advanced_stats_df, 999999) is None


def test_select_advanced_columns_normal(fake_advanced_stats_df):
    row = metrics.extract_player_advanced_row(fake_advanced_stats_df, 2544)
    result = metrics.select_advanced_columns(row)
    assert result == {"TS%": 0.62, "USG%": 0.31, "PACE": 98.5, "PIE": 0.155}


def test_select_advanced_columns_none_input():
    result = metrics.select_advanced_columns(None)
    assert result == {"TS%": None, "USG%": None, "PACE": None, "PIE": None}


def test_build_comparison_table_shape_and_values():
    table = metrics.build_comparison_table(
        {
            "LeBron James": {"TS%": 0.62, "PIE": 0.155},
            "Kevin Durant": {"TS%": 0.64, "PIE": 0.148},
        }
    )
    assert list(table.columns) == ["LeBron James", "Kevin Durant"]
    assert list(table.index) == ["TS%", "PIE"]
    assert table.loc["TS%", "Kevin Durant"] == pytest.approx(0.64)


def test_season_averages_are_per_game_means():
    log = pd.DataFrame({"PTS": [20, 30], "REB": [5, 7], "AST": [8, 9]})
    assert metrics.season_averages(log) == {"PTS": 25.0, "REB": 6.0, "AST": 8.5}
