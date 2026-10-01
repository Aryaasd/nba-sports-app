"""One-time pull of 2025-26 NBA contracts from the BALLDONTLIE API into a dated snapshot.

    python -m scripts.fetch_contracts

Needs BALLDONTLIE_API_KEY (a GOAT-tier key) in the environment or in a .env file at the
repo root. BALLDONTLIE's terms allow publishing and modeling this data; the snapshot is
committed so neither the app nor retraining ever needs the key again.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from insights import contract_value as cv

BASE_URL = "https://api.balldontlie.io/v1"
SEASON_START = 2025  # BALLDONTLIE numbers seasons by start year: 2025 = 2025-26
OUTPUT = cv.CONTRACT_DIR / f"contracts_{cv.CONTRACT_SEASON}.csv"
# Contract type only matters for possible rookie-scale deals: first-round picks still
# inside the four-year scale. Looking up only them keeps a trial key's 5 requests/min viable.
ROOKIE_SCALE_DRAFT_YEARS = range(SEASON_START - cv.ROOKIE_SCALE_SEASONS + 1, SEASON_START + 1)


def _api_key() -> str:
    key = os.environ.get("BALLDONTLIE_API_KEY")
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "BALLDONTLIE_API_KEY":
                key = value.strip()
    if not key:
        sys.exit("Set BALLDONTLIE_API_KEY in the environment or in .env at the repo root.")
    return key


def _wait_for_rate_limit_reset(response: requests.Response) -> None:
    reset_at = float(response.headers.get("x-ratelimit-reset", time.time() + 60))
    time.sleep(max(reset_at - time.time(), 1) + 1)


def _get(session: requests.Session, path: str, **params) -> list[dict]:
    """Paced by the API's own rate-limit headers, so any tier's limit is respected."""
    for _ in range(6):
        response = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if response.status_code == 429:
            _wait_for_rate_limit_reset(response)
            continue
        response.raise_for_status()
        if response.headers.get("x-ratelimit-remaining") == "0":
            _wait_for_rate_limit_reset(response)
        return response.json()["data"]
    raise RuntimeError(f"Rate-limited repeatedly on {path}")


def _contract_covering_season(aggregates: list[dict]) -> dict:
    for contract in aggregates:
        if contract["start_year"] <= SEASON_START <= contract["end_year"]:
            return contract
    return {}


def main() -> None:
    session = requests.Session()
    session.headers["Authorization"] = _api_key()

    # /teams includes defunct franchises; current ones are the 30 with a conference.
    teams = [t for t in _get(session, "/teams") if t["conference"] in ("East", "West")]
    if len(teams) != 30:
        sys.exit(f"Expected 30 current teams, got {len(teams)}")

    rows = []
    for team in teams:
        for row in _get(session, "/contracts/teams", team_id=team["id"], season=SEASON_START):
            player = row["player"]
            rows.append(
                {
                    "bdl_player_id": player["id"],
                    "PLAYER_NAME": f"{player['first_name']} {player['last_name']}",
                    "TEAM": team["abbreviation"],
                    "CAP_HIT": row["cap_hit"],
                    "TOTAL_CASH": row["total_cash"],
                    "BASE_SALARY": row["base_salary"],
                    "DRAFT_YEAR": player.get("draft_year"),
                    "DRAFT_ROUND": player.get("draft_round"),
                    "DRAFT_NUMBER": player.get("draft_number"),
                }
            )
    contracts = pd.DataFrame(rows)
    print(f"{len(contracts)} team-contract rows for {contracts['bdl_player_id'].nunique()} players")

    candidates = contracts[
        (contracts["DRAFT_ROUND"] == 1) & contracts["DRAFT_YEAR"].isin(ROOKIE_SCALE_DRAFT_YEARS)
    ]["bdl_player_id"].unique()
    print(f"Looking up contract type for {len(candidates)} possible rookie-scale players")
    details = []
    for i, player_id in enumerate(candidates, 1):
        aggregates = _get(session, "/contracts/players/aggregate", player_id=player_id)
        contract = _contract_covering_season(aggregates)
        details.append(
            {
                "bdl_player_id": player_id,
                "CONTRACT_TYPE": contract.get("contract_type"),
                "SIGNED_USING": contract.get("signed_using"),
                "CONTRACT_START": contract.get("start_year"),
                "CONTRACT_END": contract.get("end_year"),
                "CONTRACT_TOTAL_VALUE": contract.get("total_value"),
            }
        )
        if i % 20 == 0:
            print(f"  contract details: {i}/{len(candidates)}", flush=True)
    contracts = contracts.merge(pd.DataFrame(details), on="bdl_player_id", how="left")

    cv.CONTRACT_DIR.mkdir(exist_ok=True)
    contracts.sort_values(["TEAM", "CAP_HIT"], ascending=[True, False]).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({date.today().isoformat()})")


if __name__ == "__main__":
    main()
