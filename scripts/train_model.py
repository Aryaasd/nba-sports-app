"""Train, evaluate, and export the next-game points model.

Run from the repo root:  .venv/bin/python -m scripts.train_model

Trains on one full season with date-block cross-validation, then scores on the entire
following season, which the model never saw in any form. The Ridge is what ships (to
model/next_game_points.json, read by the app via modeling.load_linear_model); the random
forest is fit only to show whether a nonlinear model buys anything over it. The verdict is
driven by player-bootstrap confidence intervals, not point estimates, and is printed as-is
either way: "doesn't beat the baseline" is a legitimate result.
"""
from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np
import sklearn

import data
import model_training
import modeling

TRAIN_SEASON = "2024-25"
HOLDOUT_SEASON = "2025-26"
N_SPLITS = 5
JSON_FLOAT_DIGITS = 6

MODEL_LABELS = {
    "baseline_roll5": "Baseline (trailing 5-game PTS avg)",
    "baseline_roll10": "Baseline (trailing 10-game PTS avg)",
    "ridge": f"Ridge (alpha={model_training.RIDGE_ALPHA:g})",
    "random_forest": "Random forest",
}


def _round_floats(obj, ndigits: int = JSON_FLOAT_DIGITS):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    lines = ["| Model | MAE | RMSE |", "|---|---:|---:|"]
    for name, label in MODEL_LABELS.items():
        lines.append(f"| {label} | {metrics[name]['mae']:.3f} | {metrics[name]['rmse']:.3f} |")
    return "\n".join(lines)


def _folds_table(cv: dict) -> str:
    lines = [
        "| Fold | Validation dates | Train rows | Val rows | 5-game MAE | 10-game MAE | Ridge MAE | RF MAE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for f in cv["folds"]:
        maes = " | ".join(f"{f['metrics'][name]['mae']:.3f}" for name in MODEL_LABELS)
        dates = f"{f['val_start']} to {f['val_end']}"
        lines.append(f"| {f['fold']} | {dates} | {f['n_train']} | {f['n_val']} | {maes} |")
    return "\n".join(lines)


def _outcome(comparison: dict[str, float]) -> str:
    """"better" / "worse" only when the whole CI clears zero; a CI that includes zero is a tie."""
    if comparison["ci_low"] > 0:
        return "better"
    if comparison["ci_high"] < 0:
        return "worse"
    return "tie"


def _comparisons_table(comparisons: dict[str, dict[str, float]]) -> str:
    results = {"better": "Ridge better", "worse": "Ridge worse", "tie": "tie (CI includes 0)"}
    lines = [
        f"| Ridge vs. | MAE improvement (pts) | {model_training.BOOTSTRAP_LEVEL:.0%} CI | Result |",
        "|---|---:|---|---|",
    ]
    for name, c in comparisons.items():
        lines.append(
            f"| {MODEL_LABELS[name]} | {c['mae_improvement']:+.3f} "
            f"| [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {results[_outcome(c)]} |"
        )
    return "\n".join(lines)


def _verdict(comparisons: dict[str, dict[str, float]]) -> str:
    phrases = {"better": "beats", "worse": "is worse than", "tie": "is statistically indistinguishable from"}
    short, long = comparisons["baseline_roll5"], comparisons["baseline_roll10"]
    connector = "and" if _outcome(short) == _outcome(long) else "but"

    def detail(c):
        level = model_training.BOOTSTRAP_LEVEL
        return f"{c['mae_improvement']:+.3f} pts, {level:.0%} CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"

    tails = {
        "better": "the features add real signal beyond a longer window.",
        "worse": "a plain 10-game average is more accurate, so the model doesn't earn its complexity.",
        "tie": "the features add nothing measurable beyond a longer window.",
    }
    return (
        f"Verdict: on the unseen {HOLDOUT_SEASON} season, Ridge {phrases[_outcome(short)]} the 5-game "
        f"average ({detail(short)}) {connector} {phrases[_outcome(long)]} a plain 10-game average "
        f"({detail(long)}) -- {tails[_outcome(long)]}"
    )


def main() -> None:
    started = time.perf_counter()

    print(f"Fetching {TRAIN_SEASON} league game logs...")
    train_features = modeling.engineer_features(data.fetch_league_game_logs(TRAIN_SEASON))
    X_train = train_features[modeling.FEATURE_COLUMNS]
    y_train = train_features[modeling.TARGET_COLUMN]
    print(f"  {len(train_features)} training rows with full feature history")

    print(f"Running {N_SPLITS}-fold date-block time-series CV...")
    cv = model_training.evaluate_models_cv(train_features, n_splits=N_SPLITS)

    ridge = model_training.build_ridge_pipeline().fit(X_train, y_train)
    forest = model_training.build_random_forest().fit(X_train, y_train)

    print(f"Fetching {HOLDOUT_SEASON} league game logs (holdout)...")
    holdout_features = modeling.engineer_features(data.fetch_league_game_logs(HOLDOUT_SEASON))
    print(f"  {len(holdout_features)} holdout rows")
    holdout = model_training.evaluate_on_holdout({"ridge": ridge, "random_forest": forest}, holdout_features)
    holdout_X = holdout_features[modeling.FEATURE_COLUMNS]
    comparisons = model_training.compare_to_baselines(ridge.predict(holdout_X), holdout_features)

    exported = model_training.export_linear_model(
        ridge,
        modeling.FEATURE_COLUMNS,
        {
            "training_season": TRAIN_SEASON,
            "holdout_season": HOLDOUT_SEASON,
            "n_train_rows": len(train_features),
            "n_holdout_rows": len(holdout_features),
            "ridge_alpha": model_training.RIDGE_ALPHA,
            "cv_n_splits": N_SPLITS,
            "cv_metrics": cv,
            "holdout_metrics": holdout,
            "holdout_comparisons": comparisons,
            "bootstrap": dict(model_training.BOOTSTRAP_SETTINGS),
            "sklearn_version": sklearn.__version__,
            "created": dt.date.today().isoformat(),
        },
    )
    exported = _round_floats(exported)
    # The file that ships must reproduce the model that was just evaluated, rounding included.
    max_gap = np.max(np.abs(modeling.predict_linear(exported, holdout_X) - ridge.predict(holdout_X)))
    if max_gap > 1e-3:
        raise RuntimeError(f"Exported model drifts from the fitted pipeline by {max_gap:.2e} points")

    modeling.DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    modeling.DEFAULT_MODEL_PATH.write_text(json.dumps(exported, indent=2) + "\n")

    print(f"\n### {N_SPLITS}-fold time-series CV on {TRAIN_SEASON} (mean across folds)\n")
    print(_metrics_table(cv["mean"]))
    print(f"\n### CV folds (MAE)\n\n{_folds_table(cv)}")
    print(f"\n### Holdout: full {HOLDOUT_SEASON} season, never seen in training\n")
    print(_metrics_table(holdout))
    print(
        f"\nRidge's holdout MAE improvement over each baseline, with a player-clustered bootstrap CI "
        f"({model_training.BOOTSTRAP_N} resamples of whole players):\n"
    )
    print(_comparisons_table(comparisons))
    print(f"\n{_verdict(comparisons)}")
    print(f"\nWrote {modeling.DEFAULT_MODEL_PATH} in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
