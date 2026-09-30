import json

import numpy as np
import pandas as pd
import pytest

import model_training
import modeling


@pytest.fixture
def shuffled_dates():
    """Several rows per date (many players, same night), in shuffled row order."""
    rng = np.random.default_rng(0)
    nights = pd.date_range("2024-10-22", periods=30, freq="D")
    dates = pd.Series(np.repeat(nights, rng.integers(2, 7, size=len(nights))))
    return dates.iloc[rng.permutation(len(dates))].reset_index(drop=True)


def _synthetic_features(n_dates=60, players_per_date=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_dates * players_per_date
    df = pd.DataFrame(
        rng.normal(loc=15, scale=5, size=(n, len(modeling.FEATURE_COLUMNS))), columns=modeling.FEATURE_COLUMNS
    )
    df["GAME_DATE"] = np.repeat(pd.date_range("2024-10-22", periods=n_dates, freq="D"), players_per_date)
    df["PTS"] = 0.6 * df["PTS_roll10"] + 0.3 * df["PTS_roll5"] + rng.normal(scale=4, size=n)
    return df


def test_cv_splits_never_train_on_or_after_validation_dates(shuffled_dates):
    splits = model_training.time_ordered_cv_splits(shuffled_dates, n_splits=5)
    assert len(splits) == 5
    for train, val in splits:
        assert len(train) and len(val)
        assert shuffled_dates.iloc[train].max() < shuffled_dates.iloc[val].min()


def test_cv_splits_cover_each_date_in_at_most_one_validation_fold(shuffled_dates):
    seen_val_dates: set = set()
    for _, val in model_training.time_ordered_cv_splits(shuffled_dates, n_splits=5):
        fold_dates = set(shuffled_dates.iloc[val])
        assert not fold_dates & seen_val_dates
        seen_val_dates |= fold_dates
        # A validation date takes every row on that date -- no night is cut in half.
        assert set(val) == set(np.flatnonzero(shuffled_dates.isin(fold_dates)))


def test_exported_coefficients_reproduce_pipeline_predictions():
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.5, 20, size=len(modeling.FEATURE_COLUMNS))
    means = rng.uniform(-5, 30, size=len(modeling.FEATURE_COLUMNS))

    def random_features(n):
        values = means + scales * rng.normal(size=(n, len(means)))
        return pd.DataFrame(values, columns=modeling.FEATURE_COLUMNS)

    X_train = random_features(200)
    y_train = X_train.to_numpy() @ rng.normal(size=len(means)) + rng.normal(size=200)
    pipeline = model_training.build_ridge_pipeline(alpha=3.0).fit(X_train, y_train)

    exported = model_training.export_linear_model(
        pipeline, modeling.FEATURE_COLUMNS, {"training_season": "2024-25", "ridge_alpha": 3.0}
    )
    X_new = random_features(50)

    assert np.allclose(modeling.predict_linear(exported, X_new), pipeline.predict(X_new))
    assert exported["training_season"] == "2024-25"
    assert list(exported["standardized_coefficients"]) == modeling.FEATURE_COLUMNS
    json.dumps(exported)  # plain floats/str only


def test_evaluate_models_cv_returns_metrics_for_all_models():
    result = model_training.evaluate_models_cv(_synthetic_features(), n_splits=3)

    assert len(result["folds"]) == 3
    assert set(model_training.MODEL_NAMES) == {"baseline_roll5", "baseline_roll10", "ridge", "random_forest"}
    for fold in result["folds"]:
        assert set(fold["metrics"]) == set(model_training.MODEL_NAMES)
    for name in model_training.MODEL_NAMES:
        for metric in ("mae", "rmse"):
            value = result["mean"][name][metric]
            assert isinstance(value, float) and np.isfinite(value) and value >= 0
    json.dumps(result)


def test_evaluate_on_holdout_always_includes_both_baselines():
    train, holdout = _synthetic_features(seed=0), _synthetic_features(seed=1)
    ridge = model_training.build_ridge_pipeline().fit(train[modeling.FEATURE_COLUMNS], train["PTS"])

    result = model_training.evaluate_on_holdout({"ridge": ridge}, holdout)

    assert set(result) == {"baseline_roll5", "baseline_roll10", "ridge"}
    for name, column in [
        ("baseline_roll5", modeling.BASELINE_COLUMN),
        ("baseline_roll10", modeling.STRONG_BASELINE_COLUMN),
    ]:
        assert result[name]["mae"] == pytest.approx(np.mean(np.abs(holdout["PTS"] - holdout[column])))
