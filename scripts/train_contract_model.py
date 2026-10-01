"""Build the contract-value results the app reads (insights/contract_value.py).

    python -m scripts.train_contract_model

Joins the 2025-26 contracts snapshot (scripts/fetch_contracts.py) to that season's
stats, cross-validates the candidate models, scores every player out-of-fold, and writes
contract_data/contract_values_*.parquet plus a JSON summary. Offline only; the deployed
app just reads the results.
"""
from __future__ import annotations

import json
import sys
from datetime import date

import pandas as pd
import sklearn

import contract_training as ct
import data
import sample_data_loader
from insights import contract_value as cv

CONTRACTS_CSV = cv.CONTRACT_DIR / f"contracts_{cv.CONTRACT_SEASON}.csv"
SOURCE = "BALLDONTLIE API contracts endpoints (balldontlie.io), pulled 2026-10-01"
# Salary-source spelling -> nba_api spelling, for names normalize_name can't reconcile
# (nicknames, name order). Every remaining unmatched name is a player who didn't play.
ALIASES = {
    "Airious Bailey": "Ace Bailey",
    "Alexandre Sarr": "Alex Sarr",
    "Carlton Carrington": "Bub Carrington",
    "Hansen Yang": "Yang Hansen",
    "Nah'Shon Hyland": "Bones Hyland",
    "Nicolas Claxton": "Nic Claxton",
    "Nigel Hayes": "Nigel Hayes-Davis",
}

BIO_SNAPSHOT = cv.CONTRACT_DIR / f"player_bio_{cv.CONTRACT_SEASON}.parquet"
STAT_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "USG_PCT", "TS_PCT", "PIE",
]
OUTPUT_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "SALARY", "IMPLIED_SALARY",
    "RATIO", "LABEL", "ROOKIE_SCALE", "MINIMUM_DEAL", "NEAR_MAX", "PARTIAL_DEAL", "QUALIFIED", "IN_POOL",
    "CENSORING",
]


def _season_stats() -> pd.DataFrame:
    if not BIO_SNAPSHOT.exists():
        data.fetch_player_bio_stats(cv.CONTRACT_SEASON).to_parquet(BIO_SNAPSHOT, index=False)
    bio = pd.read_parquet(BIO_SNAPSHOT)
    advanced = sample_data_loader.load_advanced_stats(cv.CONTRACT_SEASON)[STAT_COLUMNS]
    return advanced.merge(bio, on="PLAYER_ID", how="inner")


def main() -> None:
    if not CONTRACTS_CSV.exists():
        sys.exit(f"No contracts snapshot at {CONTRACTS_CSV} -- run python -m scripts.fetch_contracts first.")

    salaries = ct.load_contracts(CONTRACTS_CSV)
    stats = _season_stats()
    joined, unmatched = cv.join_salaries(stats, salaries, ALIASES)
    print(f"Joined {len(joined)} of {len(salaries)} salary rows to {cv.CONTRACT_SEASON} stats.")
    if unmatched:
        print(f"{len(unmatched)} salary names matched no player (didn't play, or need an alias):")
        print("  " + ", ".join(unmatched))
    # The other direction: players with minutes but no contract in the snapshot (BALLDONTLIE
    # omits two-way deals, and players waived mid-season keep only their final contract).
    no_contract = stats[(stats["GP"] > 0) & ~stats["PLAYER_ID"].isin(joined["PLAYER_ID"])]
    qualified_no_contract = sum(cv.qualifies(g, m) for g, m in zip(no_contract["GP"], no_contract["MIN"]))
    print(
        f"{len(no_contract)} players with minutes have no contract on file; "
        f"{qualified_no_contract} of them qualify."
    )

    players = ct.add_flags_and_features(joined)
    pool = players[players["IN_POOL"]]
    n_max = int(pool["NEAR_MAX"].sum())
    n_market = len(pool) - n_max
    metrics = ct.evaluate_models(pool)

    print(
        f"\n### {ct.N_SPLITS}-fold CV x {ct.N_REPEATS} repeats on {n_market} market-priced "
        f"contracts (+{n_max} max deals in training)\n"
    )
    print("| Model | Typical miss | Median % error | R² | Max deals priced well below |")
    print("|---|---:|---:|---:|---:|")
    for name, m in metrics.items():
        print(
            f"| {name} | {m['typical_ratio']:.2f}x | {m['median_abs_pct_error']:.0%} | {m['r2_log']:.2f} "
            f"| {m['max_deals_priced_below']:.0%} |"
        )

    # Ship whatever is most accurate on market-priced contracts -- even the minutes-only
    # baseline, if the richer models can't beat it.
    shipped = min(metrics, key=lambda name: metrics[name]["mae_log"])
    typical_log_error = metrics[shipped]["mae_log"]
    values = ct.build_contract_values(players, shipped, typical_log_error)
    values[OUTPUT_COLUMNS].to_parquet(cv.VALUES_PATH, index=False)

    summary = {
        "season": cv.CONTRACT_SEASON,
        "salary_cap": cv.SALARY_CAP,
        "source": SOURCE,
        "n_salary_rows": len(salaries),
        "n_joined": len(joined),
        "n_unmatched": len(unmatched),
        "n_players_with_minutes_without_contract": len(no_contract),
        "n_qualified_without_contract": int(qualified_no_contract),
        "n_pool": len(pool),
        "n_market_priced": n_market,
        "n_max_deals": n_max,
        "cv_splits": ct.N_SPLITS,
        "cv_repeats": ct.N_REPEATS,
        "cv_metrics": metrics,
        "shipped_model": shipped,
        "typical_log_error": typical_log_error,
        "label_counts": values["LABEL"].value_counts().to_dict(),
        "sklearn_version": sklearn.__version__,
        "created": date.today().isoformat(),
    }
    cv.MODEL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nShipped: {shipped} (typical miss {metrics[shipped]['typical_ratio']:.2f}x). Labels:")
    for label, count in summary["label_counts"].items():
        print(f"  {label}: {count}")
    judged = values[values["LABEL"].isin([cv.OVERPAID, cv.UNDERPAID])].sort_values("RATIO")
    for title, rows in [("Most below production", judged.head(5)), ("Most above production", judged.tail(5))]:
        print(f"\n{title}:")
        for row in rows.itertuples():
            print(
                f"  {row.PLAYER_NAME}: paid {cv.format_millions(row.SALARY)}, "
                f"production implies {cv.format_millions(row.IMPLIED_SALARY)} ({row.RATIO:.2f}x)"
            )


if __name__ == "__main__":
    main()
