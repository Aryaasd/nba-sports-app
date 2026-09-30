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
    df["PLAYER_ID"] = np.tile(np.arange(players_per_date), n_dates)
    df["PTS"] = 0.6 * df["PTS_roll10"] + 0.3 * df["PTS_roll5"] + rng.normal(scale=4, size=n)
    return df


def _player_level_points(n_players=60, games_per_player=20, seed=0):
    """(y, groups, rng): points with a per-player level, so a player's games are correlated."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_players), games_per_player)
    y = rng.normal(12, 6, size=n_players)[groups] + rng.normal(0, 5, size=len(groups))
    return y, groups, rng


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


def test_bootstrap_is_deterministic_for_a_seed():
    y, groups, rng = _player_level_points()
    model, baseline = y + rng.normal(0, 4, len(y)), y + rng.normal(0, 4.5, len(y))

    first = model_training.bootstrap_mae_improvement(y, model, baseline, groups, n_boot=500, seed=7)
    again = model_training.bootstrap_mae_improvement(y, model, baseline, groups, n_boot=500, seed=7)
    other = model_training.bootstrap_mae_improvement(y, model, baseline, groups, n_boot=500, seed=8)

    assert first == again
    assert (first["ci_low"], first["ci_high"]) != (other["ci_low"], other["ci_high"])


def test_bootstrap_ci_contains_point_estimate():
    y, groups, rng = _player_level_points()
    model, baseline = y + rng.normal(0, 4, len(y)), y + rng.normal(0, 4.2, len(y))

    result = model_training.bootstrap_mae_improvement(y, model, baseline, groups)

    expected = np.mean(np.abs(y - baseline)) - np.mean(np.abs(y - model))
    assert result["mae_improvement"] == pytest.approx(expected)
    assert result["ci_low"] <= result["mae_improvement"] <= result["ci_high"]
    assert set(result) == {"mae_improvement", "ci_low", "ci_high"}


def test_bootstrap_identical_model_is_zero_with_zero_width_ci():
    y, groups, rng = _player_level_points()
    same = y + rng.normal(0, 4, len(y))

    result = model_training.bootstrap_mae_improvement(y, same, same.copy(), groups)

    assert result == {"mae_improvement": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_bootstrap_clearly_better_model_has_ci_above_zero():
    y, groups, rng = _player_level_points()
    model, baseline = y + rng.normal(0, 1, len(y)), y + rng.normal(0, 5, len(y))

    result = model_training.bootstrap_mae_improvement(y, model, baseline, groups)

    assert result["mae_improvement"] > 0
    assert result["ci_low"] > 0


def test_bootstrap_resamples_whole_players_not_rows():
    # With a single player, every cluster resample is that same player, so the interval
    # collapses to the point estimate. A row-level bootstrap would spread it out.
    y, _, rng = _player_level_points(n_players=1, games_per_player=200)
    model, baseline = y + rng.normal(0, 2, len(y)), y + rng.normal(0, 4, len(y))

    result = model_training.bootstrap_mae_improvement(y, model, baseline, np.zeros(len(y)))

    assert result["ci_low"] == pytest.approx(result["mae_improvement"])
    assert result["ci_high"] == pytest.approx(result["mae_improvement"])


def test_export_carries_holdout_comparisons_and_bootstrap_settings():
    features = _synthetic_features()
    X, y = features[modeling.FEATURE_COLUMNS], features["PTS"]
    pipeline = model_training.build_ridge_pipeline().fit(X, y)
    comparisons = model_training.compare_to_baselines(pipeline.predict(X), features)

    exported = model_training.export_linear_model(
        pipeline,
        modeling.FEATURE_COLUMNS,
        {"holdout_comparisons": comparisons, "bootstrap": dict(model_training.BOOTSTRAP_SETTINGS)},
    )
    round_tripped = json.loads(json.dumps(exported))

    assert set(round_tripped["holdout_comparisons"]) == {"baseline_roll5", "baseline_roll10"}
    for comparison in round_tripped["holdout_comparisons"].values():
        assert set(comparison) == {"mae_improvement", "ci_low", "ci_high"}
    assert round_tripped["bootstrap"] == {"n_boot": 2000, "unit": "player", "seed": 42, "level": 0.95}


def test_shipped_model_json_matches_ui_contract():
    model = modeling.load_linear_model()

    assert model["feature_columns"] == modeling.FEATURE_COLUMNS
    assert len(model["coefficients"]) == len(modeling.FEATURE_COLUMNS)
    assert set(model["holdout_comparisons"]) == {"baseline_roll5", "baseline_roll10"}
    for comparison in model["holdout_comparisons"].values():
        assert set(comparison) == {"mae_improvement", "ci_low", "ci_high"}
        assert comparison["ci_low"] <= comparison["mae_improvement"] <= comparison["ci_high"]
    assert model["bootstrap"] == model_training.BOOTSTRAP_SETTINGS
    assert set(model["holdout_metrics"]) == set(model_training.MODEL_NAMES)
