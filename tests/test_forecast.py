import numpy as np
import pandas as pd
import pytest

from insights import forecast


def _game_log(values, seed=None) -> pd.DataFrame:
    n = len(values)
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.date_range("2024-10-22", periods=n, freq="2D"),
            "MATCHUP": ["LAL vs. BOS"] * n,
            "PTS": values,
            "REB": [7] * n,
            "AST": [8] * n,
            "MIN": [35.0] * n,
        }
    )
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def test_walk_forward_never_uses_data_at_or_after_its_origin(monkeypatch):
    series = np.array([float(i * i) for i in range(15)])  # all distinct, so an off-by-one can't hide
    min_train = 5
    received = {method: [] for method in forecast.METHODS}

    def recording_stub(method, offset):
        def stub(history):
            received[method].append(np.array(history, copy=True))
            return offset + len(history)

        return stub

    # Distinct offsets per method so a swapped column would also fail.
    naive_stub = recording_stub("naive", 0.0)
    mean_stub = recording_stub("season_mean", 100.0)
    ses_stub = recording_stub("ses", 200.0)
    monkeypatch.setattr(forecast, "naive_forecast", naive_stub)
    monkeypatch.setattr(forecast, "season_mean_forecast", mean_stub)
    monkeypatch.setattr(forecast, "ses_forecast", lambda h: (ses_stub(h), 0.5))

    result = forecast.walk_forward_errors(series, min_train_size=min_train)

    origins = list(range(min_train, len(series)))
    for method in forecast.METHODS:
        histories = received[method]
        assert [len(h) for h in histories] == origins
        for k, history in zip(origins, histories):
            np.testing.assert_array_equal(history, series[:k])

    predictions = result["predictions"]
    assert list(predictions["origin"]) == origins
    np.testing.assert_array_equal(predictions["actual"], series[min_train:])
    for method, offset in [("naive", 0.0), ("season_mean", 100.0), ("ses", 200.0)]:
        expected_forecasts = offset + np.array(origins, dtype=float)
        np.testing.assert_array_equal(predictions[method], expected_forecasts)
        expected_mae = np.mean(np.abs(series[min_train:] - expected_forecasts))
        assert result["metrics"][method]["mae"] == pytest.approx(expected_mae)


def test_naive_is_last_value():
    assert forecast.naive_forecast([3, 9, 4]) == 4.0
    assert forecast.naive_forecast(pd.Series([10, 20, 30], index=[7, 3, 5])) == 30.0


def test_season_mean_is_expanding_mean():
    rng = np.random.default_rng(0)
    series = rng.normal(25, 7, 30)
    for k in range(1, len(series) + 1):
        assert forecast.season_mean_forecast(series[:k]) == pytest.approx(series[:k].mean())

    predictions = forecast.walk_forward_errors(series, min_train_size=10)["predictions"]
    expected = pd.Series(series).expanding().mean().shift(1).iloc[10:].to_numpy()
    np.testing.assert_allclose(predictions["season_mean"], expected)


def test_ses_alpha_in_unit_interval():
    for seed in range(5):
        series = np.random.default_rng(seed).poisson(22, 40)
        value, alpha = forecast.ses_forecast(series)
        assert 0.0 <= alpha <= 1.0
        assert np.isfinite(value)


def test_ses_constant_series_does_not_crash():
    assert forecast.ses_forecast([7.0] * 12) == (7.0, 0.0)
    assert forecast.ses_forecast([0] * 12) == (0.0, 0.0)
    assert forecast.ses_forecast([5.0]) == (5.0, 0.0)

    metrics = forecast.walk_forward_errors([7.0] * 25)["metrics"]
    for method in forecast.METHODS:
        assert metrics[method] == {"mae": 0.0, "rmse": 0.0}


def test_ses_close_to_naive_on_random_walk_and_close_to_mean_on_iid_noise():
    # Loose bounds: across 200 seeds the fitted alpha never fell below 0.64 on the
    # random walk or rose above 0.18 on iid noise.
    rng = np.random.default_rng(7)
    step_sd, noise_sd = 3.0, 7.0

    random_walk = 25 + np.cumsum(rng.normal(0, step_sd, 80))
    value, alpha = forecast.ses_forecast(random_walk)
    assert alpha > 0.5
    assert abs(value - random_walk[-1]) < step_sd

    iid_noise = rng.normal(25, noise_sd, 80)
    value, alpha = forecast.ses_forecast(iid_noise)
    assert alpha < 0.3
    assert abs(value - iid_noise.mean()) < noise_sd


def test_summarize_forecast_insufficient_data_guard():
    too_few = forecast.MIN_TOTAL_GAMES - 1
    result = forecast.summarize_forecast(_game_log(list(range(too_few))))
    assert result["insufficient_data"] is True
    assert result["n_games"] == too_few
    assert result["stat_col"] == "PTS"
    assert "metrics" not in result
    assert "next_forecast" not in result

    enough = forecast.summarize_forecast(_game_log(list(range(forecast.MIN_TOTAL_GAMES))))
    assert enough["insufficient_data"] is False


def test_summarize_forecast_chart_df_shape_and_series_labels():
    n = 30
    points = np.random.default_rng(3).poisson(24, n).astype(float)
    game_log = _game_log(points, seed=11)  # shuffled rows: summarize must re-sort by date
    result = forecast.summarize_forecast(game_log, stat_col="PTS")
    chart_df = result["chart_df"]

    assert list(chart_df.columns) == ["GAME_DATE", "Series", "Value"]
    assert set(chart_df["Series"]) == {"Actual", "SES forecast", "Season-mean baseline"}

    dates = pd.date_range("2024-10-22", periods=n, freq="2D")
    actual = chart_df[chart_df["Series"] == "Actual"]
    assert list(actual["GAME_DATE"]) == list(dates)
    np.testing.assert_array_equal(actual["Value"], points)

    n_forecasts = n - forecast.MIN_TRAIN_SIZE
    ses = chart_df[chart_df["Series"] == "SES forecast"]
    baseline = chart_df[chart_df["Series"] == "Season-mean baseline"]
    for series_df in (ses, baseline):
        assert len(series_df) == n_forecasts
        assert list(series_df["GAME_DATE"]) == list(dates[forecast.MIN_TRAIN_SIZE :])

    # Each forecast sits on the date of the game it predicted, built from games before it.
    k = forecast.MIN_TRAIN_SIZE
    assert ses["Value"].iloc[0] == pytest.approx(forecast.ses_forecast(points[:k])[0])
    assert baseline["Value"].iloc[-1] == pytest.approx(points[:-1].mean())


def test_metrics_finite_for_all_methods():
    points = np.random.default_rng(5).poisson(26, 70).astype(float)
    result = forecast.summarize_forecast(_game_log(points))

    assert set(result["metrics"]) == set(forecast.METHODS)
    for scores in result["metrics"].values():
        assert np.isfinite(scores["mae"]) and np.isfinite(scores["rmse"])
        assert scores["rmse"] >= scores["mae"] >= 0.0
    assert np.isfinite(result["next_forecast"])
    assert 0.0 <= result["alpha"] <= 1.0


def test_describe_alpha_bands():
    low = forecast.describe_alpha(0.05)
    assert "season-long average" in low and "barely move" in low
    assert "under 1%" in forecast.describe_alpha(0.0)

    assert "Blends season average and recent form" in forecast.describe_alpha(0.3)
    assert "Chases the most recent games" in forecast.describe_alpha(0.9)

    assert forecast.describe_alpha(forecast.ALPHA_LOW).startswith("Blends")
    assert forecast.describe_alpha(forecast.ALPHA_HIGH).startswith("Chases")
