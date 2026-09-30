# Bundled sample data

A small, real snapshot the app falls back to when the live NBA stats API is
unreachable (stats.nba.com blocks many cloud hosts, including Streamlit
Community Cloud). The UI labels it clearly whenever it's in use, and any
selection outside it shows a normal error instead of made-up data.

| | |
|---|---|
| Players | LeBron James (2544), Stephen Curry (201939) |
| Season | 2025-26 regular season (completed, so the numbers never change) |
| Captured | 2026-09-30 via `python -m scripts.capture_sample_data` |
| Source | `nba_api` 1.10.2 endpoints: PlayerGameLog, ShotChartDetail, TeamGameLog, LeagueDashPlayerStats, LeagueDashTeamStats |

Per-player files cover only those two players. The league-wide tables
(`advanced_stats`, `team_advanced_stats`) are complete, since the defense-tier
insight needs every team's rating to compute its league median.

Each file is the live fetcher's post-processed output, stored as Parquet so
dtypes survive the round trip exactly.
