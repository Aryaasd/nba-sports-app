"""Hot Hand Fallacy test: a permutation test, not a dashboard metric.

Real academic debate -- Gilovich, Vallone & Tversky (1985) argued the "hot hand"
in basketball is a cognitive illusion; Miller & Sanjurjo (2015) later showed the
original test itself was subtly biased (conditioning on a streak within a finite
sequence systematically underestimates the true conditional probability). This
sidesteps hand-deriving that correction by using a permutation test instead:
shuffle each game's own shot sequence many times and see how often shuffled
data produces streakiness at least as extreme as what was actually observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TOTAL_SHOTS = 200
MIN_GROUP_SHOTS = 20


def order_shots_within_game(shot_df: pd.DataFrame) -> pd.DataFrame:
    """Chronological shot order within each game.

    Sorted by GAME_ID, then PERIOD ascending, then remaining game-clock time
    descending -- clock time counts *down* within a period, so descending
    remaining time is ascending chronological order. Getting this backwards
    silently scrambles the sequence rather than raising any error, which is
    why it's called out explicitly here instead of left implicit.
    """
    df = shot_df.copy()
    df["_clock_seconds"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    ordered = df.sort_values(
        ["GAME_ID", "PERIOD", "_clock_seconds"], ascending=[True, True, False]
    )
    return ordered.drop(columns="_clock_seconds").reset_index(drop=True)


def extract_consecutive_pairs(ordered_shot_df: pd.DataFrame) -> pd.DataFrame:
    """(prev_made, curr_made) for every shot after the first in its game.

    A game's first shot has no valid "previous shot" and is excluded -- it
    must never be paired with the last shot of a different game.
    """
    df = ordered_shot_df
    prev_made = df["SHOT_MADE_FLAG"].shift(1)
    same_game = df["GAME_ID"] == df["GAME_ID"].shift(1)
    return pd.DataFrame(
        {
            "prev_made": prev_made[same_game].astype(int),
            "curr_made": df["SHOT_MADE_FLAG"][same_game].astype(int),
        }
    ).reset_index(drop=True)


def conditional_make_rates(pairs_df: pd.DataFrame) -> dict:
    """P(make | prev make) vs. P(make | prev miss), plus each group's sample size."""
    after_make = pairs_df.loc[pairs_df["prev_made"] == 1, "curr_made"]
    after_miss = pairs_df.loc[pairs_df["prev_made"] == 0, "curr_made"]
    return {
        "p_make_after_make": after_make.mean() if len(after_make) else float("nan"),
        "p_make_after_miss": after_miss.mean() if len(after_miss) else float("nan"),
        "n_after_make": len(after_make),
        "n_after_miss": len(after_miss),
    }


def _pooled_diff(pairs_df: pd.DataFrame) -> float:
    rates = conditional_make_rates(pairs_df)
    return rates["p_make_after_make"] - rates["p_make_after_miss"]


def permutation_test(shot_df: pd.DataFrame, n_permutations: int = 2000, seed: int = 42) -> dict:
    """Null distribution of the conditional-rate-difference statistic.

    Built by shuffling each game's own make/miss sequence independently (never
    mixing shots across games, always preserving each game's own make count),
    recomputing the pooled statistic after every shuffle.
    """
    ordered = order_shots_within_game(shot_df)
    observed_diff = _pooled_diff(extract_consecutive_pairs(ordered))

    rng = np.random.default_rng(seed)
    game_flags = [g["SHOT_MADE_FLAG"].to_numpy() for _, g in ordered.groupby("GAME_ID", sort=False)]

    null_diffs = np.empty(n_permutations)
    shuffled_df = ordered.copy()
    for i in range(n_permutations):
        shuffled_df["SHOT_MADE_FLAG"] = np.concatenate([rng.permutation(flags) for flags in game_flags])
        null_diffs[i] = _pooled_diff(extract_consecutive_pairs(shuffled_df))

    p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))

    return {
        "observed_diff": observed_diff,
        "null_distribution": null_diffs,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def summarize_hot_hand(shot_df: pd.DataFrame, n_permutations: int = 2000, seed: int = 42) -> dict:
    """Full analysis, orchestrated, with a small-sample guard.

    Returns `insufficient_data=True` rather than a p-value computed from too
    few shots to say anything meaningful.
    """
    total_shots = len(shot_df)
    ordered = order_shots_within_game(shot_df)
    rates = conditional_make_rates(extract_consecutive_pairs(ordered))

    if (
        total_shots < MIN_TOTAL_SHOTS
        or rates["n_after_make"] < MIN_GROUP_SHOTS
        or rates["n_after_miss"] < MIN_GROUP_SHOTS
    ):
        return {"insufficient_data": True, "total_shots": total_shots}

    test_result = permutation_test(shot_df, n_permutations=n_permutations, seed=seed)
    return {"insufficient_data": False, "total_shots": total_shots, **rates, **test_result}
