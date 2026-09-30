"""Next-game points model: feature engineering and inference, shared by training and the live app.

Training (model_training.py, scripts/train_model.py) and serving (the Streamlit app)
build features through these same functions, so the two can't silently drift apart.
The shipped model is a Ridge regression exported as plain JSON coefficients, which is
why this module needs only pandas/numpy -- no sklearn, streamlit, or nba_api -- and the
deployed host never has to install scikit-learn.

Every feature for a game uses only games strictly before it: rolling means are shifted
one game, computed per player, and require a full window. A model that sees its own
target in its inputs looks great in evaluation and is useless in practice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from insights import defense_tiers

ROLL_WINDOWS = (5, 10)
ROLL_STATS = ("PTS", "REB", "AST", "MIN")
FEATURE_COLUMNS = [f"{stat}_roll{w}" for w in ROLL_WINDOWS for stat in ROLL_STATS] + ["DAYS_REST", "IS_HOME"]
TARGET_COLUMN = "PTS"
BASELINE_COLUMN = "PTS_roll5"
# A plain 10-game average: the bar the model's features have to clear beyond a longer window.
STRONG_BASELINE_COLUMN = "PTS_roll10"
MAX_DAYS_REST = 7

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "next_game_points.json"


def _sort_by_player_and_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing w-game means of ROLL_STATS per player, excluding the current game."""
    df = _sort_by_player_and_date(df).copy()
    by_player = df.groupby("PLAYER_ID")
    for w in ROLL_WINDOWS:
        for stat in ROLL_STATS:
            # groupby: histories never blend across players. shift(1): a game is never in its
            # own features. min_periods=w: no 1-2 game "averages" early in a season.
            df[f"{stat}_roll{w}"] = by_player[stat].transform(
                lambda s: s.astype(float).rolling(w, min_periods=w).mean().shift(1)
            )
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """DAYS_REST since the player's previous game (back-to-back = 0, clipped) and IS_HOME."""
    df = _sort_by_player_and_date(df).copy()
    gap_days = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    # Clipped so a multi-week injury absence doesn't dominate a linear coefficient.
    df["DAYS_REST"] = (gap_days - 1).clip(0, MAX_DAYS_REST)
    df["IS_HOME"] = df["MATCHUP"].map(defense_tiers.is_home_game).astype(int)
    return df


def engineer_features(game_log_df: pd.DataFrame) -> pd.DataFrame:
    """Model-ready rows: identifiers, FEATURE_COLUMNS, and the PTS target.

    Rows without a full feature history (each player's first 10 games) or without a target
    are dropped.
    """
    if "PLAYER_ID" not in game_log_df.columns:
        raise ValueError("engineer_features needs a PLAYER_ID column (use df.assign(PLAYER_ID=...))")
    output_columns = ["PLAYER_ID", "GAME_DATE", "MATCHUP", *FEATURE_COLUMNS, TARGET_COLUMN]
    # An empty log arrives with GAME_DATE unparsed (parse_and_sort_game_log skips empties).
    if game_log_df.empty:
        return pd.DataFrame(columns=output_columns)
    # Two rows on one player-date would put one game's stats in the other's "previous games"
    # window. The real data has none, so fail loudly instead of guessing which row to keep.
    duplicated = game_log_df.duplicated(["PLAYER_ID", "GAME_DATE"], keep=False)
    if duplicated.any():
        player_ids = sorted(game_log_df.loc[duplicated, "PLAYER_ID"].unique().tolist())
        raise ValueError(f"Duplicate (PLAYER_ID, GAME_DATE) rows for PLAYER_ID(s): {player_ids}")
    df = add_rest_features(add_rolling_features(game_log_df))
    df = df.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])[output_columns]
    return df.sort_values(["GAME_DATE", "PLAYER_ID"], kind="mergesort").reset_index(drop=True)


def baseline_predictions(features_df: pd.DataFrame, column: str = BASELINE_COLUMN) -> pd.Series:
    """Naive baseline: the player's trailing points average (5-game by default)."""
    return features_df[column]


def load_linear_model(path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text())


def predict_linear(model: dict, features_df: pd.DataFrame) -> np.ndarray:
    """intercept + X @ coefficients, with X's columns taken in model["feature_columns"] order."""
    X = features_df[list(model["feature_columns"])].to_numpy(dtype=float)
    return model["intercept"] + X @ np.asarray(model["coefficients"], dtype=float)


def build_next_game_row(game_log_df: pd.DataFrame, days_rest: int, is_home: bool) -> pd.DataFrame | None:
    """One row of FEATURE_COLUMNS for the not-yet-played game after the last one in the log.

    No shift here: the next game comes after every logged game, so each rolling feature is
    the mean of the last w played games -- exactly what engineer_features would compute for
    that game once it's appended. None if the log is shorter than the longest window, or if
    any game in that window has a missing stat -- training requires full windows, so serving
    must never average 4 games and call it a 5-game average.
    """
    if len(game_log_df) < max(ROLL_WINDOWS):
        return None
    log = game_log_df.sort_values("GAME_DATE", kind="mergesort")
    if log[list(ROLL_STATS)].iloc[-max(ROLL_WINDOWS) :].isna().any().any():
        return None
    row = {
        f"{stat}_roll{w}": float(log[stat].astype(float).iloc[-w:].mean())
        for w in ROLL_WINDOWS
        for stat in ROLL_STATS
    }
    row["DAYS_REST"] = float(np.clip(days_rest, 0, MAX_DAYS_REST))
    row["IS_HOME"] = int(is_home)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def player_backtest(game_log_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Actual points vs. the one-step-ahead Model and both baseline predictions for each game
    with a full feature history. `game_log_df` is one player's log with PLAYER_ID assigned."""
    features = engineer_features(game_log_df)
    return pd.DataFrame(
        {
            "GAME_DATE": features["GAME_DATE"],
            "Actual": features[TARGET_COLUMN].astype(float),
            "Model": predict_linear(model, features),
            "Baseline (5-game avg)": baseline_predictions(features, BASELINE_COLUMN),
            "Baseline (10-game avg)": baseline_predictions(features, STRONG_BASELINE_COLUMN),
        }
    )


def describe_comparison(comparison: dict) -> str:
    """Verdict phrase for one entry of the model JSON's `holdout_comparisons`.

    Driven by the bootstrap CI, not the point estimate: an interval that includes zero
    is a tie however the point estimate leans.
    """
    gain, low, high = comparison["mae_improvement"], comparison["ci_low"], comparison["ci_high"]
    interval = f"95% CI {low:+.3f} to {high:+.3f}"
    if low > 0:
        return f"is better by {gain:.2f} pts ({interval})"
    if high < 0:
        return f"is worse by {-gain:.2f} pts ({interval})"
    return f"is statistically tied ({gain:+.3f} pts, {interval} includes zero)"
