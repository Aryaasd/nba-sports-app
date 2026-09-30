"""League-wide walk-forward comparison of the season forecasters (numbers for the README).

    python -m scripts.evaluate_forecast

Runs insights.forecast on every player with at least MIN_GAMES games in SEASON and
reports how often exponential smoothing beats each baseline, so the README's claim
rests on hundreds of player-seasons rather than one hand-picked example.
"""
from __future__ import annotations

import pandas as pd

import data
from insights import forecast

SEASON = "2025-26"
MIN_GAMES = 40
STATS = ("PTS", "REB", "AST")


def main() -> None:
    league = data.fetch_league_game_logs(SEASON)
    counts = league.groupby("PLAYER_ID").size()
    player_ids = counts[counts >= MIN_GAMES].index

    rows = []
    for player_id in player_ids:
        log = league[league["PLAYER_ID"] == player_id]
        for stat in STATS:
            result = forecast.summarize_forecast(log, stat)
            m = result["metrics"]
            rows.append(
                {
                    "stat": stat,
                    "alpha": result["alpha"],
                    **{f"{method}_mae": m[method]["mae"] for method in forecast.METHODS},
                }
            )
    df = pd.DataFrame(rows)

    print(f"### Walk-forward forecast, {SEASON}: {len(player_ids)} players with {MIN_GAMES}+ games\n")
    print(
        "| Stat | Smoothing MAE | Season-mean MAE | Last-game MAE | Smoothing beats season mean | Median α |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for stat, group in df.groupby("stat", sort=False):
        beats = (group["ses_mae"] < group["season_mean_mae"]).mean()
        print(
            f"| {stat} | {group['ses_mae'].mean():.2f} | {group['season_mean_mae'].mean():.2f} "
            f"| {group['naive_mae'].mean():.2f} | {beats:.0%} of players | {group['alpha'].median():.2f} |"
        )


if __name__ == "__main__":
    main()
