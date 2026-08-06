import numpy as np
import pandas as pd
import pytest

from insights import hot_hand


def test_order_shots_within_game_sorts_chronologically():
    df = pd.DataFrame(
        {
            "GAME_ID": ["G2", "G1", "G1", "G2", "G1"],
            "PERIOD": [1, 2, 1, 1, 1],
            "MINUTES_REMAINING": [5, 3, 10, 2, 8],
            "SECONDS_REMAINING": [0, 0, 0, 0, 0],
            "SHOT_MADE_FLAG": [1, 0, 1, 0, 1],
        }
    )
    result = hot_hand.order_shots_within_game(df)
    g1 = result[result["GAME_ID"] == "G1"]
    # period 1 shots first (descending clock: 10 before 8), then period 2.
    assert list(g1["MINUTES_REMAINING"]) == [10, 8, 3]
    assert list(g1["PERIOD"]) == [1, 1, 2]


def test_extract_consecutive_pairs_never_crosses_game_boundary():
    df = pd.DataFrame(
        {
            "GAME_ID": ["G1", "G1", "G2", "G2"],
            "PERIOD": [1, 1, 1, 1],
            "MINUTES_REMAINING": [10, 5, 10, 5],
            "SECONDS_REMAINING": [0, 0, 0, 0],
            "SHOT_MADE_FLAG": [1, 0, 1, 1],
        }
    )
    ordered = hot_hand.order_shots_within_game(df)
    pairs = hot_hand.extract_consecutive_pairs(ordered)
    # 4 shots across 2 games -> exactly 1 valid pair per game = 2 total,
    # never 3 (which is what a boundary-crossing bug would produce).
    assert len(pairs) == 2
    assert set(zip(pairs["prev_made"], pairs["curr_made"])) == {(1, 0), (1, 1)}


def test_conditional_make_rates_known_values():
    pairs = pd.DataFrame(
        {
            "prev_made": [1, 1, 1, 0, 0],
            "curr_made": [1, 1, 0, 0, 1],
        }
    )
    rates = hot_hand.conditional_make_rates(pairs)
    assert rates["n_after_make"] == 3
    assert rates["n_after_miss"] == 2
    assert rates["p_make_after_make"] == pytest.approx(2 / 3)
    assert rates["p_make_after_miss"] == pytest.approx(1 / 2)


def _synthetic_shots(n_games=10, shots_per_game=15, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        for s in range(shots_per_game):
            rows.append(
                {
                    "GAME_ID": f"G{g:03d}",
                    "PERIOD": s // 4 + 1,
                    "MINUTES_REMAINING": int(rng.integers(0, 12)),
                    "SECONDS_REMAINING": int(rng.integers(0, 60)),
                    "SHOT_MADE_FLAG": int(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def test_permutation_test_deterministic_with_seed():
    df = _synthetic_shots()
    r1 = hot_hand.permutation_test(df, n_permutations=200, seed=42)
    r2 = hot_hand.permutation_test(df, n_permutations=200, seed=42)
    assert r1["p_value"] == r2["p_value"]
    assert r1["observed_diff"] == pytest.approx(r2["observed_diff"])
    assert np.array_equal(r1["null_distribution"], r2["null_distribution"])
    assert len(r1["null_distribution"]) == 200


def test_summarize_hot_hand_insufficient_data_guard():
    tiny_df = pd.DataFrame(
        {
            "GAME_ID": ["G1"] * 5,
            "PERIOD": [1] * 5,
            "MINUTES_REMAINING": [10, 8, 6, 4, 2],
            "SECONDS_REMAINING": [0] * 5,
            "SHOT_MADE_FLAG": [1, 0, 1, 0, 1],
        }
    )
    result = hot_hand.summarize_hot_hand(tiny_df)
    assert result["insufficient_data"] is True
    assert "p_value" not in result


def test_summarize_hot_hand_runs_full_test_above_threshold():
    df = _synthetic_shots(n_games=20, shots_per_game=15)  # 300 shots, clears both guards
    result = hot_hand.summarize_hot_hand(df, n_permutations=200)
    assert result["insufficient_data"] is False
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_after_make"] >= hot_hand.MIN_GROUP_SHOTS
    assert result["n_after_miss"] >= hot_hand.MIN_GROUP_SHOTS
