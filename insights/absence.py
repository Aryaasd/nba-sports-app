"""Absence & Rest Impact: games missed and return-game performance.

Honest reframe of "health statistics" -- no free live injury-designation feed
exists anywhere (not an nba_api gap, nobody publishes real injury type/severity
for free), so this only ever measures *that* a game was missed, never *why*.

Known limitation: assumes the player stayed on one team all season. A mid-season
trade will misregister old-team games as "missed." Not worth engineering around
for this scope -- the UI states this caveat rather than hiding it.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_STAT_COLS = ["PTS", "REB", "AST", "MIN"]
MIN_ABSENCE_EVENTS = 2


def find_missed_games(team_game_dates: list, player_game_dates: list) -> list:
    """Dates the team played that don't appear in the player's own game log."""
    played = set(player_game_dates)
    return sorted(d for d in team_game_dates if d not in played)


def identify_absence_gaps(team_game_dates: list, missed_dates: list) -> list[list]:
    """Group missed dates into absence events: maximal runs of consecutive
    misses in the team's own schedule order.

    Adjacency in `missed_dates` alone isn't enough -- a played game the player
    suited up for could sit between two missed dates without appearing in
    `missed_dates` at all, so grouping needs the full schedule for context.
    """
    schedule = sorted(team_game_dates)
    missed_set = set(missed_dates)
    gaps: list[list] = []
    current: list = []
    for d in schedule:
        if d in missed_set:
            current.append(d)
        elif current:
            gaps.append(current)
            current = []
    if current:
        gaps.append(current)
    return gaps


def compute_return_game_impact(
    player_game_log: pd.DataFrame,
    absence_gaps: list[list],
    lookback_n: int = 3,
    stat_cols: list[str] | None = None,
) -> dict:
    """Average of each absence event's next `lookback_n` games vs. the season
    average excluding those return-window games.

    Returns `insufficient_data=True` instead of a number when there are too
    few absence events (< MIN_ABSENCE_EVENTS) to say anything meaningful --
    a single gap produces a "delta" that's really just noise from one player-week.
    """
    stat_cols = stat_cols or DEFAULT_STAT_COLS

    if len(absence_gaps) < MIN_ABSENCE_EVENTS:
        return {"insufficient_data": True, "n_absence_events": len(absence_gaps)}

    df = player_game_log.sort_values("GAME_DATE").reset_index(drop=True)

    return_game_dates = set()
    for gap in absence_gaps:
        gap_end = max(gap)
        after = df[df["GAME_DATE"] > gap_end].head(lookback_n)
        return_game_dates.update(after["GAME_DATE"])

    return_games = df[df["GAME_DATE"].isin(return_game_dates)]
    baseline_games = df[~df["GAME_DATE"].isin(return_game_dates)]

    if return_games.empty or baseline_games.empty:
        return {"insufficient_data": True, "n_absence_events": len(absence_gaps)}

    stats = {}
    for col in stat_cols:
        if col not in df.columns:
            continue
        return_avg = return_games[col].mean()
        baseline_avg = baseline_games[col].mean()
        stats[col] = {
            "return_avg": return_avg,
            "baseline_avg": baseline_avg,
            "delta": return_avg - baseline_avg,
        }

    return {
        "insufficient_data": False,
        "n_absence_events": len(absence_gaps),
        "n_return_games": len(return_games),
        "stats": stats,
    }
