"""Contract value: is a player paid more or less than his production implies?

Honest framing, because "overpaid" is easy to overclaim:

- The model learns what the market *pays for* (minutes, scoring volume, efficiency,
  age), not what wins games. A gap means "priced differently from the market", not
  "bad contract".
- Contracts price past and expected future value; this compares them with one season.
- Rookie-scale deals (first-round picks in their first four seasons) are set by the
  league's pay scale, not negotiated, so they're valued but never called "underpaid".
- Near-max deals are capped by the CBA: if production implies more, the cap is why.
- Every player's value comes from a model that never saw his own salary (out-of-fold),
  and a gap within the model's typical error is reported as fairly paid.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_SEASON = "2025-26"
SEASON_START_YEAR = 2025
SALARY_CAP = 154_647_000  # 2025-26, per the NBA's official announcement
# Max salary is 25/30/35% of the cap by years of service; at or above 95% of the lowest
# tier counts as on (or near) a max deal. Nobody can sign for more than 35%.
NEAR_MAX_SALARY = 0.25 * SALARY_CAP * 0.95
HIGHEST_MAX_SALARY = 0.35 * SALARY_CAP
# 2025-26 minimum salary (0 years of service). A cap hit below it is a prorated fragment
# of a ten-day or rest-of-season deal, not an annual salary.
LEAGUE_MINIMUM_SALARY = 1_272_870
# Top of the 2025-26 minimum-salary scale (10+ years of service). The scale runs from
# $1,272,870 to this by experience, and multi-year minimum deals add small raises, so at or
# below it pay sits at the league floor rather than at a market price. (Every scale amount
# checked -- e.g. Melton's $3,080,921 at 7 years -- appears to the dollar in the data.)
NEAR_MINIMUM_SALARY = 3_634_153
ROOKIE_SCALE_SEASONS = 4
MIN_GAMES = 20
MIN_MINUTES_PER_GAME = 10.0

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "contract_data"
VALUES_PATH = CONTRACT_DIR / f"contract_values_{CONTRACT_SEASON}.parquet"
MODEL_SUMMARY_PATH = CONTRACT_DIR / "contract_model.json"

MODEL_LABELS = {
    "baseline_minutes_only": "minutes-only linear model",
    "ridge": "Ridge regression",
    "gradient_boosting": "gradient-boosted tree model",
}

FAIR = "Fairly paid"
OVERPAID = "Paid above production"
UNDERPAID = "Paid below production"
ROOKIE_SCALE = "Rookie-scale contract"
MINIMUM_DEAL = "Minimum-level contract"
MAX_DEAL = "Max contract"
PARTIAL_DEAL = "Partial-season deal"
NOT_ENOUGH_MINUTES = "Not enough minutes to judge"
# Labels that come with a production-implied dollar figure worth showing.
VALUED_LABELS = {FAIR, OVERPAID, UNDERPAID, ROOKIE_SCALE, MINIMUM_DEAL, MAX_DEAL}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Join key for names across sources: "Nikola Jokić" -> "nikola jokic",
    "Jaren Jackson Jr." -> "jaren jackson", "A.J. Lawson" -> "aj lawson"."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[.'’]", "", ascii_name.lower()).replace("-", " ").split()
    return " ".join(t for t in tokens if t not in _SUFFIXES)


def join_salaries(
    players_df: pd.DataFrame, salaries_df: pd.DataFrame, aliases: dict[str, str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Attach salary columns to players_df (PLAYER_NAME, ...) by normalized name.

    `salaries_df` has PLAYER_NAME, SALARY, and any other columns to carry across (e.g. an
    exact ROOKIE_SCALE flag). `aliases` maps a salary-source name to the
    nba_api spelling for names normalization can't reconcile (e.g. nicknames). Returns the
    joined frame (players without a salary dropped) and the salary-source names that
    matched nobody, so the join rate is visible rather than silently lossy.
    """
    aliases = aliases or {}
    salaries = salaries_df.assign(
        _key=salaries_df["PLAYER_NAME"].map(lambda n: normalize_name(aliases.get(n, n)))
    )
    duplicated = salaries["_key"].duplicated(keep=False)
    if duplicated.any():
        names = sorted(salaries.loc[duplicated, "PLAYER_NAME"])
        raise ValueError(f"Salary source lists these players more than once: {names}")

    players = players_df.assign(_key=players_df["PLAYER_NAME"].map(normalize_name))
    joined = players.merge(salaries.drop(columns="PLAYER_NAME"), on="_key", how="inner")
    unmatched = sorted(salaries.loc[~salaries["_key"].isin(players["_key"]), "PLAYER_NAME"])
    return joined.drop(columns="_key"), unmatched


def qualifies(games: float, minutes_per_game: float) -> bool:
    return games >= MIN_GAMES and minutes_per_game >= MIN_MINUTES_PER_GAME


def verdict(
    actual: float,
    implied: float,
    typical_log_error: float,
    *,
    qualified: bool,
    rookie_scale: bool = False,
    partial_deal: bool = False,
    minimum_deal: bool = False,
    near_max: bool = False,
) -> dict:
    """Label plus the gap as a ratio (actual / implied).

    Only market-priced contracts get a market verdict. Rookie-scale pay is set by the
    league's scale, minimum deals sit on a price floor, and a max player whose production
    prices above his salary is held there by the cap's ceiling -- none of those gaps say
    anything about the market. Otherwise, within `typical_log_error` (the model's
    cross-validated mean absolute error in log salary) counts as fair.
    """
    if partial_deal:
        return {"label": PARTIAL_DEAL, "ratio": None}
    if not qualified:
        return {"label": NOT_ENOUGH_MINUTES, "ratio": None}
    ratio = actual / implied
    if rookie_scale:
        return {"label": ROOKIE_SCALE, "ratio": ratio}
    if minimum_deal:
        return {"label": MINIMUM_DEAL, "ratio": ratio}
    if near_max and implied > actual:
        return {"label": MAX_DEAL, "ratio": ratio}
    if abs(np.log(ratio)) <= typical_log_error:
        return {"label": FAIR, "ratio": ratio}
    return {"label": OVERPAID if ratio > 1 else UNDERPAID, "ratio": ratio}


def format_millions(dollars: float) -> str:
    return f"${dollars / 1e6:.1f}M"


def describe_verdict(result: dict, implied: float) -> str:
    """One plain-language sentence for the UI."""
    label, ratio = result["label"], result["ratio"]
    reference = f"For reference, his production prices at about {format_millions(implied)}."
    if label == PARTIAL_DEAL:
        return (
            "On a ten-day or rest-of-season deal, so his cap hit is a fraction of a season's "
            "pay -- no verdict."
        )
    if label == NOT_ENOUGH_MINUTES:
        return (
            f"Played under {MIN_GAMES} games or {MIN_MINUTES_PER_GAME:.0f} minutes a night -- "
            "too little to judge."
        )
    if label == ROOKIE_SCALE:
        return (
            "On a rookie-scale deal, which the league's pay scale sets, not the market -- "
            f"no verdict. {reference}"
        )
    if label == MINIMUM_DEAL:
        return (
            f"Paid at the league-minimum level (at or below {format_millions(NEAR_MINIMUM_SALARY)}), "
            f"a price floor rather than a market price -- no verdict. {reference}"
        )
    if label == MAX_DEAL:
        ceiling = (
            f", more than the {format_millions(HIGHEST_MAX_SALARY)} ceiling anyone can sign for"
            if implied > HIGHEST_MAX_SALARY
            else ""
        )
        return (
            f"On or near a max deal, and his production prices higher ({format_millions(implied)}{ceiling}). "
            "The salary cap, not the market, holds his pay down -- no verdict."
        )
    if label == FAIR:
        return (
            "The gap to his production is within the model's typical error, so it can't call "
            "him over- or underpaid."
        )
    if label == OVERPAID:
        return f"Paid about {ratio:.1f}x what his production implies."
    return f"His production implies about {1 / ratio:.1f}x his salary."


def load_contract_values(path: Path = VALUES_PATH) -> pd.DataFrame | None:
    """Precomputed per-player values (scripts/train_contract_model.py), or None if not built."""
    return pd.read_parquet(path) if Path(path).exists() else None


def load_model_summary(path: Path = MODEL_SUMMARY_PATH) -> dict | None:
    """CV metrics, typical error, and data provenance written alongside the values."""
    return json.loads(Path(path).read_text()) if Path(path).exists() else None
