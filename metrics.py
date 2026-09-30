"""Pure logic for the NBA app: season strings, player lookup, stat math, advanced-metrics assembly.

No streamlit or nba_api imports here on purpose — keeps this module fast to unit test
and reusable by future scripts (e.g. a model-training script) without pulling in Streamlit.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEFAULT_STAT_COLS = ["PTS", "REB", "AST"]

ADVANCED_METRIC_LABELS = {
    "TS_PCT": "TS%",
    "USG_PCT": "USG%",
    "PACE": "PACE",
    "PIE": "PIE",
}


def get_current_season(today: dt.date | None = None) -> str:
    """Return the NBA season string (e.g. "2025-26") that is "current" as of `today`.

    NBA seasons start in October, so before October the current/most-recently-completed
    season is the one that started the previous calendar year.
    """
    today = today or dt.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def get_recent_seasons(n: int = 5, today: dt.date | None = None) -> list[str]:
    """Return the `n` most recent season strings, most-recent-first."""
    current = get_current_season(today)
    start_year = int(current[:4])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, start_year - n, -1)]


def build_player_index(all_players: list[dict]) -> dict[str, int]:
    """Map lowercased full name -> player id, for O(1) lookup."""
    return {p["full_name"].lower(): p["id"] for p in all_players}


def get_player_id(index: dict[str, int], name: str) -> int | None:
    return index.get(name.lower())


def get_player_names(all_players: list[dict]) -> list[str]:
    return sorted(p["full_name"] for p in all_players)


def build_team_abbreviation_index(all_teams: list[dict]) -> dict[int, str]:
    """Map team id -> abbreviation. Used to join nba_api endpoints (like team
    advanced stats) that key by TEAM_ID but don't include an abbreviation column,
    against MATCHUP strings elsewhere that only give the abbreviation."""
    return {t["id"]: t["abbreviation"] for t in all_teams}


def build_abbreviation_to_team_id_index(all_teams: list[dict]) -> dict[str, int]:
    """Map team abbreviation -> id, the reverse of build_team_abbreviation_index.
    Used to resolve a team_id from a MATCHUP string's abbreviation (e.g. for
    PlayerGameLog, which has no TEAM_ID column of its own to read directly)."""
    return {t["abbreviation"]: t["id"] for t in all_teams}


def safe_selectbox_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    """Index of `preferred` in `options`, or `fallback` if it isn't present."""
    try:
        return options.index(preferred)
    except ValueError:
        return fallback


def parse_and_sort_game_log(
    df: pd.DataFrame, date_col: str = "GAME_DATE", date_format: str | None = "%b %d, %Y"
) -> pd.DataFrame:
    """Parse `date_col` to datetime and return the df sorted ascending by it.

    The default format matches the singular game-log endpoints ("OCT 24, 2023");
    pass `date_format=None` for the league-wide endpoint's ISO timestamps.
    """
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    return df.sort_values(date_col).reset_index(drop=True)


def _index_of(*candidates) -> pd.Index | None:
    """Index of the first candidate that's a pd.Series, so scalar/array inputs can ride along."""
    for c in candidates:
        if isinstance(c, pd.Series):
            return c.index
    return None


def true_shooting_pct(pts, fga, fta):
    """TS% = PTS / (2 * (FGA + 0.44 * FTA)). Zero denominator -> NaN. Scalars or Series."""
    index = _index_of(pts, fga, fta)
    pts_arr = np.asarray(pts, dtype=float)
    denom = 2 * (np.asarray(fga, dtype=float) + 0.44 * np.asarray(fta, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0, np.nan, pts_arr / denom)
    if index is not None:
        return pd.Series(result, index=index)
    return float(result) if result.ndim == 0 else result


def per_36(value, minutes):
    """value/minutes*36. Zero minutes -> NaN. Scalars or Series."""
    index = _index_of(value, minutes)
    value_arr = np.asarray(value, dtype=float)
    minutes_arr = np.asarray(minutes, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(minutes_arr == 0, np.nan, value_arr / minutes_arr * 36)
    if index is not None:
        return pd.Series(result, index=index)
    return float(result) if result.ndim == 0 else result


def season_per36_totals(game_log_df: pd.DataFrame, stat_cols: list[str] | None = None) -> dict[str, float]:
    """Season per-36 rates from already-fetched game log totals (no extra API call).

    Keys are suffixed "<STAT>/36" so the result can be merged directly into a
    comparison table alongside the advanced-metric labels.
    """
    stat_cols = stat_cols or DEFAULT_STAT_COLS
    if game_log_df.empty or "MIN" not in game_log_df.columns:
        return {f"{col}/36": float("nan") for col in stat_cols}
    total_minutes = game_log_df["MIN"].sum()
    return {f"{col}/36": per_36(game_log_df[col].sum(), total_minutes) for col in stat_cols}


def extract_player_advanced_row(advanced_df: pd.DataFrame, player_id: int) -> pd.Series | None:
    """Row for `player_id` from a whole-league advanced-stats df, joined by ID not name."""
    matches = advanced_df[advanced_df["PLAYER_ID"] == player_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def select_advanced_columns(row: pd.Series | None) -> dict:
    """Map a whole advanced-stats row to just the four labeled metrics this app shows.

    `None` in -> dict of `None`s out, so the caller can render "N/A" without a branch.
    """
    if row is None:
        return dict.fromkeys(ADVANCED_METRIC_LABELS.values())
    return {label: row[col] for col, label in ADVANCED_METRIC_LABELS.items()}


def build_comparison_table(player_stats: dict[str, dict[str, float | None]]) -> pd.DataFrame:
    """player_stats: {player_name: {metric_label: value}} -> metrics-as-rows, players-as-columns."""
    return pd.DataFrame(player_stats)
