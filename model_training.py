"""Offline training and evaluation for the next-game points model (training-only: uses sklearn).

Evaluation is built to be hard to fool:
- Every model is scored against two naive baselines: the trailing 5-game and 10-game points
  averages. A model that can't beat "he'll score what he's been scoring" isn't worth shipping,
  and the 10-game average shows how much of any gain is just from using a longer window.
- Cross-validation is time-ordered and splits on whole dates. On any given night many
  players share a GAME_DATE, so a row-level split would put some of that night's games in
  training and the rest in validation; splitting on unique dates means every fold trains
  strictly on the past and validates strictly on the future.
- The scaler lives inside the pipeline, so each fold standardizes on its own training rows.
- A whole unseen season is held out for the final score, and each holdout gain over a
  baseline gets a bootstrap confidence interval that resamples whole players (one player's
  games aren't independent draws). A gain whose interval includes zero is a tie, not a win.

Only the Ridge ships. It's exported as raw-scale coefficients so the app can predict with
plain numpy (see modeling.predict_linear); the random forest is a comparison point only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import modeling

RIDGE_ALPHA = 1.0
BASELINES = {
    "baseline_roll5": modeling.BASELINE_COLUMN,
    "baseline_roll10": modeling.STRONG_BASELINE_COLUMN,
}
MODEL_NAMES = (*BASELINES, "ridge", "random_forest")

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_SETTINGS = {
    "n_boot": BOOTSTRAP_N,
    "unit": "player",
    "seed": BOOTSTRAP_SEED,
    "level": BOOTSTRAP_LEVEL,
}


def time_ordered_cv_splits(dates: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, validation) positional row indices, splitting on whole unique dates.

    TimeSeriesSplit runs over the sorted unique dates, then each date block maps back to
    every row on that date -- whatever order the rows arrive in.
    """
    date_codes, unique_dates = pd.factorize(pd.Series(dates), sort=True)
    splits = []
    for train_dates, val_dates in TimeSeriesSplit(n_splits=n_splits).split(unique_dates):
        train_rows = np.flatnonzero(np.isin(date_codes, train_dates))
        val_rows = np.flatnonzero(np.isin(date_codes, val_dates))
        splits.append((train_rows, val_rows))
    return splits


def build_ridge_pipeline(alpha: float = RIDGE_ALPHA) -> Pipeline:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1
    )


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _baseline_predictions(features_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: modeling.baseline_predictions(features_df, column) for name, column in BASELINES.items()}


def evaluate_models_cv(features_df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Per-fold and mean MAE/RMSE for both baselines, Ridge, and random forest.

    {"folds": [{"fold", "n_train", "n_val", "val_start", "val_end", "metrics": {model: {mae, rmse}}}],
     "mean": {model: {mae, rmse}}} -- plain Python types throughout.
    """
    X = features_df[modeling.FEATURE_COLUMNS]
    y = features_df[modeling.TARGET_COLUMN]
    folds = []
    for fold_number, (train_idx, val_idx) in enumerate(
        time_ordered_cv_splits(features_df["GAME_DATE"], n_splits), start=1
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        predictions = {
            **_baseline_predictions(features_df.iloc[val_idx]),
            "ridge": build_ridge_pipeline().fit(X_train, y_train).predict(X_val),
            "random_forest": build_random_forest().fit(X_train, y_train).predict(X_val),
        }
        val_dates = features_df["GAME_DATE"].iloc[val_idx]
        folds.append(
            {
                "fold": fold_number,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_start": str(pd.Timestamp(val_dates.min()).date()),
                "val_end": str(pd.Timestamp(val_dates.max()).date()),
                "metrics": {name: _regression_metrics(y_val, pred) for name, pred in predictions.items()},
            }
        )
    mean = {
        name: {
            metric: float(np.mean([fold["metrics"][name][metric] for fold in folds]))
            for metric in ("mae", "rmse")
        }
        for name in MODEL_NAMES
    }
    return {"folds": folds, "mean": mean}


def evaluate_on_holdout(fitted_models: dict, features_df: pd.DataFrame) -> dict:
    """{model: {mae, rmse}} on an unseen season, always including both naive baselines."""
    X = features_df[modeling.FEATURE_COLUMNS]
    y = features_df[modeling.TARGET_COLUMN]
    baselines = _baseline_predictions(features_df)
    results = {name: _regression_metrics(y, pred) for name, pred in baselines.items()}
    for name, model in fitted_models.items():
        results[name] = _regression_metrics(y, model.predict(X))
    return results


def bootstrap_mae_improvement(
    y_true, pred_model, pred_baseline, groups, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict[str, float]:
    """baseline MAE - model MAE (positive = model better) with a 95% percentile CI from a
    cluster bootstrap that resamples whole groups (players) with replacement."""
    y = np.asarray(y_true, dtype=float)
    baseline_abs_error = np.abs(y - np.asarray(pred_baseline, dtype=float))
    model_abs_error = np.abs(y - np.asarray(pred_model, dtype=float))
    abs_error_gap = baseline_abs_error - model_abs_error
    codes, uniques = pd.factorize(np.asarray(groups))
    gap_per_group = np.bincount(codes, weights=abs_error_gap)
    rows_per_group = np.bincount(codes).astype(float)

    # Row b of `draws` counts how often each group was picked when drawing len(uniques) groups
    # with replacement -- the whole bootstrap as two matrix-vector products, no Python loop.
    n_groups = len(uniques)
    draws = np.random.default_rng(seed).multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_boot)
    boot_improvements = (draws @ gap_per_group) / (draws @ rows_per_group)

    tail = 100 * (1 - BOOTSTRAP_LEVEL) / 2
    ci_low, ci_high = np.percentile(boot_improvements, [tail, 100 - tail])
    return {
        "mae_improvement": float(abs_error_gap.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def compare_to_baselines(model_predictions, features_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{baseline name: bootstrap_mae_improvement} for a model's predictions vs. each naive baseline,
    clustered by PLAYER_ID."""
    y = features_df[modeling.TARGET_COLUMN]
    return {
        name: bootstrap_mae_improvement(y, model_predictions, pred, features_df["PLAYER_ID"])
        for name, pred in _baseline_predictions(features_df).items()
    }


def export_linear_model(pipeline: Pipeline, feature_columns: list[str], metadata: dict) -> dict:
    """A fitted StandardScaler+Ridge pipeline as JSON-ready raw-scale coefficients.

    The scaler is folded in: coef_raw = coef / scale, intercept_raw = intercept - sum(coef * mean / scale),
    so modeling.predict_linear reproduces pipeline.predict without sklearn.
    """
    scaler, ridge = pipeline[0], pipeline[-1]
    standardized = np.asarray(ridge.coef_, dtype=float)
    raw = standardized / scaler.scale_
    intercept = float(ridge.intercept_) - float(np.sum(standardized * scaler.mean_ / scaler.scale_))
    return {
        "model_type": "ridge",
        "target": modeling.TARGET_COLUMN,
        "feature_columns": list(feature_columns),
        "intercept": intercept,
        "coefficients": [float(c) for c in raw],
        "standardized_coefficients": {
            name: float(c) for name, c in zip(feature_columns, standardized, strict=True)
        },
        **metadata,
    }
