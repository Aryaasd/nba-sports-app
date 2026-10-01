"""Training for the contract-value model -- offline only (scikit-learn/scipy), like model_training.py.

Target is log(salary): pay is roughly multiplicative in production, and log errors read as
"x times" gaps. Players are scored by out-of-fold predictions, so no player's own salary
ever informs his own valuation.

The CBA bounds salaries at both ends, and each end needed a decision, settled by
cross-validation on the market-priced contracts in between:

- Minimum deals stay out. A two-sided censored (Tobit) model that read them as "worth at
  most the minimum" collapsed (typical miss 2.17x, R^2 below zero): plenty of productive
  veterans take the minimum to join a contender, so the bound simply isn't true.
- Max deals stay in, as the prices they were. Dropping them left the model blind to what
  star production costs, and reading them as "worth at least the max" (censored_regression
  below) was less accurate on market-priced deals. A max deal was the market's price when
  signed, so a star who has declined since is genuinely paid above his current production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from insights import contract_value as cv

FEATURES = ["MIN", "GP", "PTS", "REB", "AST", "USG_PCT", "TS_PCT", "PIE", "AGE", "AGE_SQ"]
BASELINE_FEATURES = ["MIN"]
MODELS = ("baseline_minutes_only", "ridge", "censored_regression")
N_SPLITS = 5
# Out-of-fold predictions are averaged over repeated, differently shuffled CV runs: a single
# run's fold assignment alone flipped 9-15 of ~270 verdicts between seeds.
N_REPEATS = 10
SEED = 42
L2_ALPHA = 1.0

UNCENSORED, FLOOR, CEILING = 0, -1, 1


def load_contracts(path) -> pd.DataFrame:
    """PLAYER_NAME, SALARY (2025-26 cap hit), and an exact ROOKIE_SCALE flag from the
    BALLDONTLIE snapshot (scripts/fetch_contracts.py).

    Contract type was only fetched for first-round picks still inside the rookie-scale
    window -- nobody else can be on a rookie-scale deal, so a missing type means False.
    """
    raw = pd.read_csv(path)
    # BALLDONTLIE occasionally carries one person under two player ids (seen: Keaton
    # Wallace, same team and cap hit). Collapse exact copies; a same-name pair with
    # different salaries still fails loudly in join_salaries.
    raw = raw.drop_duplicates(subset=["PLAYER_NAME", "TEAM", "CAP_HIT"])
    return pd.DataFrame(
        {
            "PLAYER_NAME": raw["PLAYER_NAME"].str.strip(),
            "SALARY": raw["CAP_HIT"].astype(float),
            "ROOKIE_SCALE": raw["CONTRACT_TYPE"].eq("Rookie"),
        }
    )


def add_flags_and_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """Model features plus contract-kind flags. Needs SALARY and an exact ROOKIE_SCALE column."""
    df = players_df.copy()
    df["AGE_SQ"] = df["AGE"] ** 2
    df["QUALIFIED"] = [cv.qualifies(g, m) for g, m in zip(df["GP"], df["MIN"])]
    df["PARTIAL_DEAL"] = df["SALARY"] < cv.LEAGUE_MINIMUM_SALARY
    df["MINIMUM_DEAL"] = ~df["PARTIAL_DEAL"] & (df["SALARY"] <= cv.NEAR_MINIMUM_SALARY)
    df["NEAR_MAX"] = df["SALARY"] >= cv.NEAR_MAX_SALARY
    # Rookie-scale pay is set by the league's scale, minimum pay sits on a floor, and a
    # partial-season cap hit isn't a salary: none of them is a market price to learn from.
    df["IN_POOL"] = df["QUALIFIED"] & ~df["ROOKIE_SCALE"] & ~df["PARTIAL_DEAL"] & ~df["MINIMUM_DEAL"]
    # Only the censored model reads this; the others treat max deals as the prices they were.
    df["CENSORING"] = np.where(df["NEAR_MAX"], CEILING, UNCENSORED)
    return df


class CensoredRegression:
    """Two-sided Tobit regression with a light L2 penalty, on standardized features.

    Exactly priced rows contribute the normal density of their residual; FLOOR rows the
    probability the latent market value is at or below their salary; CEILING rows the
    probability it's at or above.
    """

    def __init__(self, alpha: float = L2_ALPHA):
        self.alpha = alpha

    def fit(self, X, y, censoring) -> CensoredRegression:
        X, y, censoring = np.asarray(X, float), np.asarray(y, float), np.asarray(censoring)
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        exact = censoring == UNCENSORED

        # Start from least squares on the exactly priced rows.
        design = np.column_stack([Z[exact], np.ones(exact.sum())])
        start, *_ = np.linalg.lstsq(design, y[exact], rcond=None)
        start_sd = max(float(np.std(y[exact] - design @ start)), 1e-3)

        def negative_log_likelihood(params):
            beta, intercept, log_sigma = params[:-2], params[-2], params[-1]
            z = (y - Z @ beta - intercept) / np.exp(log_sigma)
            log_lik = np.where(
                exact,
                norm.logpdf(z) - log_sigma,
                np.where(censoring == FLOOR, norm.logcdf(z), norm.logsf(z)),
            )
            return -log_lik.sum() + self.alpha * beta @ beta

        result = minimize(negative_log_likelihood, np.append(start, np.log(start_sd)), method="L-BFGS-B")
        self.coef_, self.intercept_, self.sigma_ = result.x[:-2], result.x[-2], float(np.exp(result.x[-1]))
        return self

    def predict(self, X) -> np.ndarray:
        return ((np.asarray(X, float) - self.mean_) / self.scale_) @ self.coef_ + self.intercept_


def model_features(name: str) -> list[str]:
    return BASELINE_FEATURES if name == "baseline_minutes_only" else FEATURES


def fit_model(name: str, X: np.ndarray, y_log: np.ndarray, censoring: np.ndarray):
    """The censored model reads max deals as lower bounds; the others as exact prices."""
    if name == "censored_regression":
        return CensoredRegression().fit(X, y_log, censoring)
    model = LinearRegression() if name == "baseline_minutes_only" else make_pipeline(
        StandardScaler(), Ridge(alpha=L2_ALPHA)
    )
    return model.fit(X, y_log)


def _arrays(name: str, pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = pool[model_features(name)].to_numpy(float)
    return X, np.log(pool["SALARY"].to_numpy(float)), pool["CENSORING"].to_numpy()


def out_of_fold_log_predictions(name: str, pool: pd.DataFrame, n_repeats: int = N_REPEATS) -> np.ndarray:
    """Each row predicted only by models fit on other folds -- never on its own salary --
    averaged over `n_repeats` differently shuffled CV runs for stable verdicts."""
    X, y_log, censoring = _arrays(name, pool)
    predictions = np.zeros(len(pool))
    for repeat in range(n_repeats):
        folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for train_idx, val_idx in folds.split(X):
            fitted = fit_model(name, X[train_idx], y_log[train_idx], censoring[train_idx])
            predictions[val_idx] += fitted.predict(X[val_idx])
    return predictions / n_repeats


def score(pool: pd.DataFrame, pred_log: np.ndarray) -> dict:
    """Accuracy on market-priced (non-max) contracts -- the shared yardstick every model is
    chosen by -- plus the share of max deals priced well below their salary."""
    y_log = np.log(pool["SALARY"].to_numpy(float))
    ceiling = pool["CENSORING"].to_numpy() == CEILING
    errors = y_log[~ceiling] - pred_log[~ceiling]
    mae = float(np.abs(errors).mean())
    r2 = 1 - (errors**2).sum() / ((y_log[~ceiling] - y_log[~ceiling].mean()) ** 2).sum()
    below = pred_log[ceiling] < y_log[ceiling] - mae
    return {
        "mae_log": mae,
        "typical_ratio": float(np.exp(mae)),
        "median_abs_pct_error": float(np.median(np.abs(np.exp(errors) - 1))),
        "r2_log": float(r2),
        "max_deals_priced_below": float(below.mean()) if ceiling.any() else 0.0,
    }


def evaluate_models(pool: pd.DataFrame) -> dict[str, dict]:
    return {name: score(pool, out_of_fold_log_predictions(name, pool)) for name in MODELS}


def build_contract_values(
    players_df: pd.DataFrame, model_name: str, typical_log_error: float
) -> pd.DataFrame:
    """Implied salary + verdict for every player: out-of-fold for the fitting pool, and a model
    fit on the whole pool for everyone outside it (rookie scale, minimum and partial deals,
    low minutes)."""
    df = players_df.copy()
    pool = df["IN_POOL"]
    implied_log = pd.Series(np.nan, index=df.index)
    implied_log[pool] = out_of_fold_log_predictions(model_name, df[pool])
    if (~pool).any():
        fitted = fit_model(model_name, *_arrays(model_name, df[pool]))
        implied_log[~pool] = fitted.predict(df.loc[~pool, model_features(model_name)].to_numpy(float))
    df["IMPLIED_SALARY"] = np.exp(implied_log)

    verdicts = [
        cv.verdict(
            row.SALARY,
            row.IMPLIED_SALARY,
            typical_log_error,
            qualified=row.QUALIFIED,
            rookie_scale=row.ROOKIE_SCALE,
            partial_deal=row.PARTIAL_DEAL,
            minimum_deal=row.MINIMUM_DEAL,
            near_max=row.NEAR_MAX,
        )
        for row in df.itertuples()
    ]
    df["LABEL"] = [v["label"] for v in verdicts]
    df["RATIO"] = [v["ratio"] for v in verdicts]
    return df
