"""Next-game forecasting: simple exponential smoothing vs. two naive baselines.

The question is modest on purpose: given a player's game-by-game stat line so
far this season, how well can next game's number be predicted, and does a real
forecasting method beat the obvious rules of thumb? Three one-step-ahead
forecasters are compared:

- naive: last game carried forward (the textbook naive forecast).
- season mean: the season-to-date average.
- SES (simple exponential smoothing): an exponentially weighted average of all
  games so far, with the weight on recent games (alpha) fitted from the data.

Why SES rather than Holt or ARIMA: within one season, per-game scoring is mostly
noise around a fairly stable talent level, with no real trend or seasonality to
model, so extra trend/AR terms mostly fit noise on 20-80 points. SES explains in
one sentence and stays robust on short series. It also sits exactly between the
two baselines: alpha -> 1 is the naive forecast, and at alpha = 0 (with an
estimated initial level) it fits one constant level, which is the season mean.
So the fitted alpha itself says something: a low alpha means recent games carry
little signal beyond the player's season level -- the same question the Hot Hand
analysis asks at the shot level, asked here at the game level.

Evaluation is rolling-origin (walk-forward), never a random split: at each game
k, every method sees only games 0..k-1 and is scored against game k. A random
split would let a forecaster train on games after the one it predicts.

Limitations, stated rather than hidden:
- Univariate. Minutes, opponent, rest, home/away, blowouts and injuries are all
  ignored, so errors are dominated by genuine game-to-game variance and the
  methods often finish close together. That is a finding, not a failure.
- One season, no prior from earlier seasons: the earliest origins train on only
  MIN_TRAIN_SIZE games, and a fitted alpha from 20-80 games is imprecise, so
  the alpha bands are a rough reading, not a measurement.
- Games are indexed by games played, not calendar time; an absence or a
  mid-season trade (a level shift) is just "the next game" to every method.
- Point forecasts only; no prediction intervals are reported.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

MIN_TOTAL_GAMES = 20
MIN_TRAIN_SIZE = 10

# describe_alpha bands. SES moves its level alpha of the way toward each new
# game's result, so below 0.15 a 10-point outlier shifts the forecast by under
# 1.5 points; from 0.5 up, a single game moves it at least halfway.
ALPHA_LOW = 0.15
ALPHA_HIGH = 0.5

METHODS = ("naive", "season_mean", "ses")
CHART_LABELS = {"ses": "SES forecast", "season_mean": "Season-mean baseline"}


def _as_history(history) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot forecast from an empty history.")
    return values


def naive_forecast(history) -> float:
    """Last observed value carried forward."""
    return float(_as_history(history)[-1])


def season_mean_forecast(history) -> float:
    """Season-to-date (expanding) mean."""
    return float(_as_history(history).mean())


def ses_forecast(history) -> tuple[float, float]:
    """(one-step-ahead forecast, fitted alpha).

    A history with no variation (including a single game) short-circuits to
    (that value, 0.0): every alpha gives the same forecast there, so alpha is
    unidentified and the optimizer would stop at an arbitrary point. 0.0 is
    reported because "the level never moved" is the honest reading.
    """
    values = _as_history(history)
    if np.ptp(values) == 0:
        return float(values[-1]), 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        fit = SimpleExpSmoothing(values, initialization_method="estimated").fit()
    return float(fit.forecast(1)[0]), float(fit.params["smoothing_level"])


def walk_forward_errors(series, min_train_size: int = MIN_TRAIN_SIZE) -> dict:
    """Rolling-origin evaluation with an expanding window.

    Origin k gives every forecaster exactly series[:k] and scores it against
    series[k]; nothing at or after the origin ever reaches a forecaster.
    """
    values = np.asarray(series, dtype=float)
    if min_train_size < 1 or len(values) <= min_train_size:
        raise ValueError(f"Need more than min_train_size={min_train_size} observations, got {len(values)}.")

    # Built per call, not at import, so tests can monkeypatch the module-level forecasters.
    forecasters = {
        "naive": naive_forecast,
        "season_mean": season_mean_forecast,
        "ses": lambda h: ses_forecast(h)[0],
    }

    rows = []
    for k in range(min_train_size, len(values)):
        history = values[:k]
        row = {"origin": k, "actual": values[k]}
        for method, forecaster in forecasters.items():
            row[method] = forecaster(history)
        rows.append(row)
    predictions = pd.DataFrame(rows, columns=["origin", "actual", *METHODS])

    metrics = {}
    for method in METHODS:
        errors = predictions["actual"] - predictions[method]
        metrics[method] = {
            "mae": float(errors.abs().mean()),
            "rmse": float(np.sqrt((errors**2).mean())),
        }
    return {"predictions": predictions, "metrics": metrics}


def summarize_forecast(game_log_df: pd.DataFrame, stat_col: str = "PTS") -> dict:
    """Walk-forward comparison plus a next-game SES forecast, with a small-sample guard.

    Returns `insufficient_data=True` (with n_games and stat_col, but no metrics)
    rather than error figures computed from too few forecasts to mean anything.
    """
    df = game_log_df.dropna(subset=[stat_col]).sort_values("GAME_DATE").reset_index(drop=True)
    n_games = len(df)
    if n_games < MIN_TOTAL_GAMES:
        return {"insufficient_data": True, "n_games": n_games, "stat_col": stat_col}

    values = df[stat_col].to_numpy(dtype=float)
    walk_forward = walk_forward_errors(values, min_train_size=MIN_TRAIN_SIZE)
    next_forecast, alpha = ses_forecast(values)

    predictions = walk_forward["predictions"]
    forecast_dates = df["GAME_DATE"].iloc[predictions["origin"].to_numpy()].to_numpy()
    frames = [pd.DataFrame({"GAME_DATE": df["GAME_DATE"], "Series": "Actual", "Value": values})]
    for method, label in CHART_LABELS.items():
        frames.append(
            pd.DataFrame(
                {"GAME_DATE": forecast_dates, "Series": label, "Value": predictions[method].to_numpy()}
            )
        )
    chart_df = pd.concat(frames, ignore_index=True)

    return {
        "insufficient_data": False,
        "n_games": n_games,
        "stat_col": stat_col,
        "metrics": walk_forward["metrics"],
        "alpha": alpha,
        "next_forecast": next_forecast,
        "chart_df": chart_df,
    }


def describe_alpha(alpha: float) -> str:
    """Plain-language reading of a fitted SES alpha (bands: ALPHA_LOW, ALPHA_HIGH)."""
    share = "under 1%" if alpha < 0.005 else f"about {alpha:.0%}"
    step = f"each new game moves it {share} of the way toward that game's result"
    if alpha < ALPHA_LOW:
        return f"Leans on the season-long average; recent games barely move the forecast ({step})."
    if alpha < ALPHA_HIGH:
        return f"Blends season average and recent form ({step})."
    return f"Chases the most recent games ({step})."
