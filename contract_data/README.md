# Contract data (2025-26)

Inputs and results for the **Contract Value** section of Compare Players.

| File | What it is | Source |
|---|---|---|
| `contracts_2025-26.csv` | Every team's standard 2025-26 contracts: cap hit, cash, base salary, and draft info. For first-round picks from the 2022-25 drafts, it also has the contract type, which tells a rookie-scale deal apart from a free-agent one. | [BALLDONTLIE API](https://www.balldontlie.io/) contracts endpoints, pulled once on 2026-10-01 with `python -m scripts.fetch_contracts` |
| `player_bio_2025-26.parquet` | Per-game PTS/REB/AST and draft year/round for every player | `nba_api` LeagueDashPlayerBioStats |
| `contract_values_2025-26.parquet` | Each player's salary, production-implied salary, and verdict or label | `python -m scripts.train_contract_model` |
| `contract_model.json` | Cross-validated metrics for every candidate model, the shipped model, and the join counts in both directions | same |

**Coverage, stated plainly:**
- BALLDONTLIE doesn't include two-way contracts.
- It lists a player waived mid-season only with his final contract.
- BALLDONTLIE lists one player (Keaton Wallace) twice under two IDs; the loader collapses those exact copies.

Of 505 contracts, 493 match a player with 2025-26 stats; the other 12 belong to players who didn't play. In the other direction, 89 players with minutes have no contract here, mostly two-way players. The app tells you when a compared player is one of them.

BALLDONTLIE's terms allow publishing and modeling its data. That's why it was chosen over salary sites whose terms forbid scraping, or bar using their data for predictive models. The app reads only the precomputed results, so the deployed version never needs an API key or a live call.
