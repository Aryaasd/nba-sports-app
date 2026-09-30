import numpy as np
import pandas as pd
import pytest

import modeling


def _game_log(pts, start="2024-10-22", gap_days=2, player_id=None, home_every=2):
    """One player's log in the live app's shape: ascending GAME_DATE, int MIN."""
    n = len(pts)
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range(start, periods=n, freq=f"{gap_days}D"),
            "MATCHUP": ["LAL vs. BOS" if i % home_every == 0 else "LAL @ BOS" for i in range(n)],
            "PTS": pts,
            "REB": [(3 * i) % 11 + 2 for i in range(n)],
            "AST": [(5 * i) % 7 + 1 for i in range(n)],
            "MIN": [28 + (7 * i) % 9 for i in range(n)],
        }
    )
    if player_id is not None:
        df["PLAYER_ID"] = player_id
    return df


PTS_A = [12, 25, 8, 30, 17, 22, 9, 28, 15, 20, 33, 11, 26, 14, 19]
PTS_B = [5, 7, 3, 9, 6, 4, 8, 2, 10, 6, 5, 7, 3, 9, 4]


def _two_players_interleaved(pts_a=PTS_A, pts_b=PTS_B):
    both = pd.concat([_game_log(pts_a, player_id=1), _game_log(pts_b, player_id=2)])
    # Same nights for both players, arriving in league-endpoint order (by date only).
    return both.sort_values("GAME_DATE", kind="mergesort").reset_index(drop=True)


def test_rolling_features_exclude_current_game():
    log = _game_log(PTS_A, player_id=1)
    result = modeling.add_rolling_features(log).reset_index(drop=True)

    assert result.loc[:4, "PTS_roll5"].isna().all()  # min_periods: no partial windows
    assert result.loc[:9, "PTS_roll10"].isna().all()
    for i in range(5, len(PTS_A)):
        assert result.loc[i, "PTS_roll5"] == pytest.approx(np.mean(PTS_A[i - 5 : i]))

    # Changing a game's own points must not change that game's features.
    bumped = list(PTS_A)
    bumped[12] = 1000
    bumped_result = modeling.add_rolling_features(_game_log(bumped, player_id=1)).reset_index(drop=True)
    features = [c for c in modeling.FEATURE_COLUMNS if "_roll" in c]
    pd.testing.assert_series_equal(bumped_result.loc[12, features], result.loc[12, features])


def test_rolling_features_no_cross_player_leakage():
    original = modeling.add_rolling_features(_two_players_interleaved())
    perturbed = modeling.add_rolling_features(_two_players_interleaved(pts_a=[p * 10 for p in PTS_A]))

    features = [c for c in modeling.FEATURE_COLUMNS if "_roll" in c]
    b_original = original[original["PLAYER_ID"] == 2].reset_index(drop=True)
    b_perturbed = perturbed[perturbed["PLAYER_ID"] == 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(b_original[features], b_perturbed[features])

    b_alone = modeling.add_rolling_features(_game_log(PTS_B, player_id=2)).reset_index(drop=True)
    pd.testing.assert_frame_equal(b_original[features], b_alone[features])


def test_days_rest_computed_from_previous_game_and_clipped():
    log = pd.DataFrame(
        {
            "PLAYER_ID": 1,
            "GAME_DATE": pd.to_datetime(["2024-10-22", "2024-10-23", "2024-10-25", "2024-11-20"]),
            "MATCHUP": ["LAL vs. BOS", "LAL @ DAL", "LAL vs. MIA", "LAL @ HOU"],
        }
    )
    result = modeling.add_rest_features(log.iloc[::-1]).reset_index(drop=True)

    assert np.isnan(result.loc[0, "DAYS_REST"])  # no previous game
    assert result.loc[1, "DAYS_REST"] == 0  # back-to-back
    assert result.loc[2, "DAYS_REST"] == 1
    assert result.loc[3, "DAYS_REST"] == modeling.MAX_DAYS_REST  # 25 days off, clipped
    assert list(result["IS_HOME"]) == [1, 0, 1, 0]


def test_engineer_features_drops_rows_without_full_history():
    log = pd.concat([_game_log(PTS_A, player_id=1), _game_log(PTS_B[:8], player_id=2)])
    result = modeling.engineer_features(log)

    assert list(result.columns) == [
        "PLAYER_ID", "GAME_DATE", "MATCHUP", *modeling.FEATURE_COLUMNS, modeling.TARGET_COLUMN
    ]
    assert len(result) == len(PTS_A) - max(modeling.ROLL_WINDOWS)  # player 2 never reaches 10 games
    assert set(result["PLAYER_ID"]) == {1}
    assert not result[modeling.FEATURE_COLUMNS].isna().any().any()
    assert isinstance(result.index, pd.RangeIndex)
    assert result["GAME_DATE"].is_monotonic_increasing


def test_engineer_features_requires_player_id():
    with pytest.raises(ValueError, match="PLAYER_ID"):
        modeling.engineer_features(_game_log(PTS_A))


@pytest.mark.parametrize("days_rest, is_home", [(0, True), (2, False), (30, True)])
def test_build_next_game_row_matches_engineer_features_for_appended_game(days_rest, is_home):
    log = _game_log(PTS_A)
    next_game = pd.DataFrame(
        {
            "GAME_DATE": [log["GAME_DATE"].iloc[-1] + pd.Timedelta(days=days_rest + 1)],
            "MATCHUP": ["LAL vs. NYK" if is_home else "LAL @ NYK"],
            "PTS": [99],
            "REB": [99],
            "AST": [99],
            "MIN": [48],
        }
    )
    appended = pd.concat([log, next_game], ignore_index=True).assign(PLAYER_ID=1)

    served = modeling.build_next_game_row(log, days_rest=days_rest, is_home=is_home)
    trained = modeling.engineer_features(appended).iloc[[-1]][modeling.FEATURE_COLUMNS]

    assert list(served.columns) == modeling.FEATURE_COLUMNS
    np.testing.assert_allclose(served.to_numpy(dtype=float), trained.to_numpy(dtype=float))


def test_build_next_game_row_none_with_too_few_games():
    needed = max(modeling.ROLL_WINDOWS)
    assert modeling.build_next_game_row(_game_log(PTS_A[: needed - 1]), days_rest=1, is_home=True) is None
    assert modeling.build_next_game_row(_game_log(PTS_A[:needed]), days_rest=1, is_home=True) is not None


def test_predict_linear_matches_manual_dot_product():
    rng = np.random.default_rng(0)
    columns = modeling.FEATURE_COLUMNS
    coefficients = rng.normal(size=len(columns))
    model = {"feature_columns": columns, "coefficients": list(coefficients), "intercept": 2.5}
    features = pd.DataFrame(rng.normal(size=(6, len(columns))), columns=columns)

    expected = 2.5 + features.to_numpy() @ coefficients
    # Column order and extra columns in the frame must not matter -- the model's order does.
    shuffled = features[features.columns[::-1]].assign(PLAYER_ID=7)
    np.testing.assert_allclose(modeling.predict_linear(model, shuffled), expected)


def test_player_backtest_columns_and_alignment():
    log = _game_log(PTS_A, player_id=1)
    coefficients = [1.0 if c == "PTS_roll10" else 0.0 for c in modeling.FEATURE_COLUMNS]
    model = {"feature_columns": modeling.FEATURE_COLUMNS, "coefficients": coefficients, "intercept": 1.0}

    result = modeling.player_backtest(log, model)

    assert list(result.columns) == [
        "GAME_DATE", "Actual", "Model", "Baseline (5-game avg)", "Baseline (10-game avg)"
    ]
    first = max(modeling.ROLL_WINDOWS)
    assert list(result["GAME_DATE"]) == list(log["GAME_DATE"].iloc[first:])
    assert list(result["Actual"]) == PTS_A[first:]
    for i, row in result.iterrows():
        game = first + i
        assert row["Model"] == pytest.approx(1.0 + np.mean(PTS_A[game - 10 : game]))
        assert row["Baseline (5-game avg)"] == pytest.approx(np.mean(PTS_A[game - 5 : game]))
        assert row["Baseline (10-game avg)"] == pytest.approx(np.mean(PTS_A[game - 10 : game]))


def test_player_backtest_empty_log_is_empty():
    empty = pd.DataFrame(columns=["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]).assign(PLAYER_ID=1)
    model = {"feature_columns": modeling.FEATURE_COLUMNS, "coefficients": [0.0] * 10, "intercept": 0.0}
    assert modeling.player_backtest(empty, model).empty
