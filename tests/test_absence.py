import pandas as pd
import pytest

from insights import absence

# 10-game schedule, spaced every 3 days. Timestamps to match what
# data.fetch_team_game_dates / data.fetch_player_game_log actually produce.
SCHEDULE = list(pd.date_range("2023-10-24", periods=10, freq="3D"))
D1, D2, D3, D4, D5, D6, D7, D8, D9, D10 = SCHEDULE

# Player missed D3 (isolated single game) and D7, D8 (a two-game absence).
PLAYED = [D1, D2, D4, D5, D6, D9, D10]
MISSED = [D3, D7, D8]


def test_find_missed_games():
    assert absence.find_missed_games(SCHEDULE, PLAYED) == MISSED


def test_identify_absence_gaps_groups_consecutive_and_splits_isolated():
    gaps = absence.identify_absence_gaps(SCHEDULE, MISSED)
    assert gaps == [[D3], [D7, D8]]


def test_identify_absence_gaps_no_misses_is_empty():
    assert absence.identify_absence_gaps(SCHEDULE, []) == []


def _player_game_log():
    return pd.DataFrame(
        {
            "GAME_DATE": PLAYED,
            "PTS": [20, 22, 25, 26, 24, 28, 27],
            "REB": [5, 6, 8, 9, 7, 10, 9],
            "AST": [4, 5, 6, 7, 5, 8, 7],
            "MIN": [30, 31, 35, 36, 34, 37, 36],
        }
    )


def test_compute_return_game_impact_known_deltas():
    gaps = absence.identify_absence_gaps(SCHEDULE, MISSED)  # [[D3], [D7, D8]]
    result = absence.compute_return_game_impact(_player_game_log(), gaps, lookback_n=3)

    assert result["insufficient_data"] is False
    assert result["n_absence_events"] == 2
    # return games: D4,D5,D6 (after D3) + D9,D10 (after D8, only 2 remain) = 5
    assert result["n_return_games"] == 5

    pts = result["stats"]["PTS"]
    assert pts["baseline_avg"] == pytest.approx((20 + 22) / 2)
    assert pts["return_avg"] == pytest.approx((25 + 26 + 24 + 28 + 27) / 5)
    assert pts["delta"] == pytest.approx(5.0)


def test_compute_return_game_impact_insufficient_data_below_threshold():
    single_gap = [[D3]]
    result = absence.compute_return_game_impact(_player_game_log(), single_gap)
    assert result["insufficient_data"] is True
    assert result["n_absence_events"] == 1
    assert "stats" not in result
