"""Snapshot real data for the bundled demo fallback (see sample_data_loader.py).

Run from the repo root on a machine stats.nba.com doesn't block (a home
connection, not a cloud host):

    python -m scripts.capture_sample_data

Writes each live fetcher's post-processed output, so a fallback returns exactly
what the live path would have. Refuses to capture a player who changed teams
mid-season, since the Absence insight assumes one team all season.
"""
from __future__ import annotations

import pandas as pd

import data
import metrics
import sample_data_loader as sdl
from insights import defense_tiers


def _single_team_abbreviation(game_log: pd.DataFrame, player_name: str) -> str:
    abbreviations = game_log["MATCHUP"].map(defense_tiers.extract_own_team_abbreviation).unique()
    if len(abbreviations) != 1:
        raise SystemExit(
            f"{player_name} played for {sorted(abbreviations)} in {sdl.SAMPLE_SEASON} -- "
            "pick a player who stayed on one team all season."
        )
    return abbreviations[0]


def main() -> None:
    season = sdl.SAMPLE_SEASON
    sdl.SAMPLE_DIR.mkdir(exist_ok=True)
    abbr_to_team_id = metrics.build_abbreviation_to_team_id_index(data._get_all_teams_uncached())

    for player_id, name in sdl.SAMPLE_PLAYERS.items():
        game_log = data._fetch_player_game_log_uncached(player_id, season)
        team_id = abbr_to_team_id[_single_team_abbreviation(game_log, name)]
        if team_id not in sdl.SAMPLE_TEAM_IDS:
            raise SystemExit(f"{name}'s team {team_id} is missing from SAMPLE_TEAM_IDS")

        game_log.to_parquet(sdl.game_log_path(player_id, season), index=False)
        data._fetch_shot_chart_uncached(player_id, season).to_parquet(
            sdl.shot_chart_path(player_id, season), index=False
        )
        team_dates = data._fetch_team_game_dates_uncached(team_id, season)
        pd.DataFrame({"GAME_DATE": team_dates}).to_parquet(
            sdl.team_game_dates_path(team_id, season), index=False
        )
        print(f"{name}: {len(game_log)} games, team {sdl.SAMPLE_TEAM_IDS[team_id]} ({len(team_dates)} dates)")

    data._fetch_advanced_stats_uncached(season).to_parquet(sdl.advanced_stats_path(season), index=False)
    data._fetch_team_advanced_stats_uncached(season).to_parquet(
        sdl.team_advanced_stats_path(season), index=False
    )
    print(f"Wrote {len(list(sdl.SAMPLE_DIR.glob('*.parquet')))} files to {sdl.SAMPLE_DIR}")


if __name__ == "__main__":
    main()
