"""Performance vs. Defensive Quality: buckets opponents by measured DEF_RATING.

Honest reframe of "different defensive schemes" -- true scheme data (zone vs.
man, coverage type) is proprietary player-tracking data (Second Spectrum) not
published anywhere for free. This answers the closest honestly-answerable
question instead: does this player perform differently against elite vs. weak
defenses, using the league's own measured DEF_RATING?
"""
from __future__ import annotations

import pandas as pd

ELITE_LABEL = "Elite Defense"
WEAK_LABEL = "Weak Defense"


def _split_matchup(matchup: str) -> tuple[str, str]:
    """(own_abbreviation, opponent_abbreviation) from e.g. "LAL vs. HOU" or "LAL @ DAL"."""
    separator = "vs." if "vs." in matchup else "@"
    own, opponent = matchup.split(separator)
    return own.strip(), opponent.strip()


def extract_opponent_abbreviation(matchup: str) -> str:
    """"LAL vs. HOU" -> "HOU", "LAL @ DAL" -> "DAL"."""
    return _split_matchup(matchup)[1]


def extract_own_team_abbreviation(matchup: str) -> str:
    """"LAL vs. HOU" -> "LAL", "LAL @ DAL" -> "LAL".

    PlayerGameLog has no TEAM_ID column at all (confirmed live) -- this is how
    the Absence & Rest Impact insight identifies the player's own team instead.
    """
    return _split_matchup(matchup)[0]


def bucket_teams_by_defense(
    team_advanced_df: pd.DataFrame, team_id_to_abbr: dict[int, str]
) -> dict[str, str]:
    """Map team abbreviation -> Elite/Weak Defense tier via a median DEF_RATING split.

    `team_id_to_abbr` comes from metrics.build_team_abbreviation_index -- team
    advanced stats are keyed by TEAM_ID with no abbreviation column of their own.
    """
    median = team_advanced_df["DEF_RATING"].median()
    tiers = {}
    for _, row in team_advanced_df.iterrows():
        abbr = team_id_to_abbr.get(row["TEAM_ID"])
        if abbr is None:
            continue
        tiers[abbr] = ELITE_LABEL if row["DEF_RATING"] <= median else WEAK_LABEL
    return tiers


def compare_performance_by_tier(
    game_log_df: pd.DataFrame, team_tier_map: dict[str, str], stat_cols: list[str] | None = None
) -> pd.DataFrame:
    """Mean per stat per opponent-defense tier, plus game count per tier so the
    reader can judge sample size themselves rather than trusting a bare average."""
    stat_cols = stat_cols or ["PTS", "REB", "AST"]
    df = game_log_df.copy()
    df["Opponent"] = df["MATCHUP"].apply(extract_opponent_abbreviation)
    df["Tier"] = df["Opponent"].map(team_tier_map)
    df = df.dropna(subset=["Tier"])
    if df.empty:
        return pd.DataFrame(columns=[*stat_cols, "Games"])
    grouped = df.groupby("Tier")[stat_cols].mean()
    grouped["Games"] = df.groupby("Tier").size()
    return grouped
