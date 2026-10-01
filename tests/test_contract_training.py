import numpy as np
import pandas as pd
import pytest

import contract_training as ct
from insights import contract_value as cv


def _players(n=120, seed=0):
    """Synthetic league: salary tracks minutes and scoring, with real floors and ceilings."""
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(5, 38, n)
    pts = minutes * rng.uniform(0.3, 0.8, n)
    salary = np.exp(13.8 + 0.08 * minutes + 0.03 * pts + rng.normal(0, 0.3, n))
    salary = np.clip(salary, cv.LEAGUE_MINIMUM_SALARY, 0.30 * cv.SALARY_CAP)
    return pd.DataFrame(
        {
            "PLAYER_ID": range(n),
            "PLAYER_NAME": [f"Player {i}" for i in range(n)],
            "MIN": minutes,
            "GP": rng.integers(5, 82, n),
            "PTS": pts,
            "REB": rng.uniform(1, 12, n),
            "AST": rng.uniform(0.5, 9, n),
            "USG_PCT": rng.uniform(0.1, 0.35, n),
            "TS_PCT": rng.uniform(0.48, 0.68, n),
            "PIE": rng.uniform(0.03, 0.2, n),
            "AGE": rng.uniform(20, 36, n),
            "ROOKIE_SCALE": rng.random(n) < 0.15,
            "SALARY": salary,
        }
    )


def _censored_sample(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    latent = 1 + 2 * x + rng.normal(0, 0.3, n)
    floor, ceiling = 1.6, 2.5
    censoring = np.where(latent <= floor, ct.FLOOR, np.where(latent >= ceiling, ct.CEILING, ct.UNCENSORED))
    return x[:, None], np.clip(latent, floor, ceiling), censoring


def test_censored_regression_recovers_the_true_slope_that_naive_fits_distort():
    X, y, censoring = _censored_sample()
    model = ct.CensoredRegression(alpha=0.0).fit(X, y, censoring)
    slope = model.predict([[1.0]])[0] - model.predict([[0.0]])[0]
    assert slope == pytest.approx(2.0, abs=0.1)
    assert model.sigma_ == pytest.approx(0.3, abs=0.03)

    exact = censoring == ct.UNCENSORED
    assert np.polyfit(X[exact, 0], y[exact], 1)[0] < 1.2  # dropping bounded rows flattens the fit
    assert np.polyfit(X[:, 0], y, 1)[0] < 1.4  # so does trusting bounds as exact prices


def test_censored_regression_matches_least_squares_without_censoring():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = X @ np.array([0.5, -1.0, 2.0]) + 3 + rng.normal(0, 0.2, 400)
    tobit = ct.CensoredRegression(alpha=0.0).fit(X, y, np.zeros(400, dtype=int))
    ols = np.linalg.lstsq(np.column_stack([X, np.ones(400)]), y, rcond=None)[0]
    np.testing.assert_allclose(tobit.predict(X), np.column_stack([X, np.ones(400)]) @ ols, atol=1e-3)


@pytest.mark.parametrize("model_name", ct.MODELS)
def test_out_of_fold_prediction_never_uses_its_own_salary(model_name):
    pool = ct.add_flags_and_features(_players())
    pool = pool[pool["IN_POOL"]].reset_index(drop=True)
    before = ct.out_of_fold_log_predictions(model_name, pool, n_repeats=2)

    leaked = pool.copy()
    leaked.loc[7, "SALARY"] *= 50  # wildly change one player's own salary
    after = ct.out_of_fold_log_predictions(model_name, leaked, n_repeats=2)

    assert after[7] == pytest.approx(before[7])  # his own valuation can't move


def test_load_contracts_uses_cap_hit_and_exact_rookie_flag(tmp_path):
    path = tmp_path / "contracts.csv"
    pd.DataFrame(
        {
            "PLAYER_NAME": ["Nikola Jokic", "Victor Wembanyama", "Jake LaRavia"],
            "TEAM": ["DEN", "SAS", "LAL"],
            "CAP_HIT": [55_224_526, 13_376_880, 6_000_000],
            "CONTRACT_TYPE": [None, "Rookie", "Free Agent"],  # LaRavia: 2022 first-rounder, not on the scale
        }
    ).to_csv(path, index=False)
    df = ct.load_contracts(path)
    assert df["SALARY"].tolist() == [55_224_526.0, 13_376_880.0, 6_000_000.0]
    assert df["ROOKIE_SCALE"].tolist() == [False, True, False]


def test_load_contracts_collapses_exact_duplicate_records_only(tmp_path):
    path = tmp_path / "contracts.csv"
    pd.DataFrame(
        {
            "PLAYER_NAME": ["Keaton Wallace", "Keaton Wallace", "Same Name", "Same Name"],
            "TEAM": ["ATL", "ATL", "BOS", "LAL"],
            "CAP_HIT": [2_296_274, 2_296_274, 3_000_000, 9_000_000],
            "CONTRACT_TYPE": [None, None, None, None],
        }
    ).to_csv(path, index=False)
    df = ct.load_contracts(path)
    assert df["PLAYER_NAME"].tolist() == ["Keaton Wallace", "Same Name", "Same Name"]


@pytest.mark.parametrize(
    "salary, in_pool, censoring",
    [
        (1_272_870, False, ct.UNCENSORED),  # rookie minimum: a floor, not a price
        (3_080_921, False, ct.UNCENSORED),  # 7-year veteran minimum
        (cv.NEAR_MINIMUM_SALARY, False, ct.UNCENSORED),  # top of the minimum scale
        (12_000_000, True, ct.UNCENSORED),
        (cv.NEAR_MAX_SALARY, True, ct.CEILING),  # a real price; a lower bound to the censored model
        (55_224_526, True, ct.CEILING),
    ],
)
def test_minimum_deals_stay_out_and_max_deals_stay_in(salary, in_pool, censoring):
    players = _players(n=1).assign(SALARY=salary, GP=60, MIN=25.0, ROOKIE_SCALE=False)
    flagged = ct.add_flags_and_features(players)
    assert flagged.loc[0, "IN_POOL"] == in_pool
    assert flagged.loc[0, "CENSORING"] == censoring


def test_rookie_scale_partial_minimum_and_low_minute_players_stay_out_of_the_fitting_pool():
    players = _players(n=4).assign(GP=60, MIN=25.0, SALARY=10e6, ROOKIE_SCALE=False)
    players.loc[0, "ROOKIE_SCALE"] = True
    players.loc[1, "SALARY"] = 131_970  # a ten-day cap hit
    players.loc[2, "SALARY"] = 2_296_274  # a veteran-minimum deal
    players.loc[3, ["GP", "MIN"]] = [8, 30.0]
    assert not ct.add_flags_and_features(players)["IN_POOL"].any()


def test_build_contract_values_scores_every_player_with_the_right_label():
    df = ct.add_flags_and_features(_players())
    values = ct.build_contract_values(df, "censored_regression", typical_log_error=0.3)
    assert values["IMPLIED_SALARY"].notna().all()
    full_season = ~values["PARTIAL_DEAL"]  # a partial deal's label takes precedence
    assert (values.loc[values["PARTIAL_DEAL"], "LABEL"] == cv.PARTIAL_DEAL).all()
    assert (values.loc[full_season & ~values["QUALIFIED"], "LABEL"] == cv.NOT_ENOUGH_MINUTES).all()
    rookies = full_season & values["ROOKIE_SCALE"] & values["QUALIFIED"]
    assert (values.loc[rookies, "LABEL"] == cv.ROOKIE_SCALE).all()
    minimums = values["QUALIFIED"] & ~values["ROOKIE_SCALE"] & values["MINIMUM_DEAL"]
    assert (values.loc[minimums, "LABEL"] == cv.MINIMUM_DEAL).all()
    market = values["IN_POOL"] & ~values["NEAR_MAX"]
    assert set(values.loc[market, "LABEL"]) <= {cv.FAIR, cv.OVERPAID, cv.UNDERPAID}


def test_evaluate_models_reports_accuracy_and_max_deal_check_for_every_model():
    df = ct.add_flags_and_features(_players(n=200))
    metrics = ct.evaluate_models(df[df["IN_POOL"]])
    assert set(metrics) == set(ct.MODELS)
    for scores in metrics.values():
        assert scores["typical_ratio"] == pytest.approx(np.exp(scores["mae_log"]))
        assert 0 <= scores["max_deals_priced_below"] <= 1
