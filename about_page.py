"""The About page: what the app does, where its numbers come from, and why it's built this way.

Static text plus a results table read from the committed model files, so it renders even
when the live NBA API is unreachable.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import modeling
import ui
from insights import contract_value, forecast, shot_chart

REPO_URL = "https://github.com/Aryaasd/nba-sports-app"

SOURCES = [
    (
        "Live stats",
        "Game logs, advanced stats, shot charts, and team ratings come from NBA.com through the "
        "open-source nba_api package. Each request is cached for an hour.",
    ),
    (
        "Backup snapshot",
        "NBA.com blocks most cloud servers, including this app's host. When it can't be reached, "
        "the app uses a saved copy of real 2025-26 data for LeBron James and Stephen Curry, with a "
        "banner saying so. Any other selection shows an error, never invented numbers.",
    ),
    (
        "Contracts",
        "2025-26 salaries from the BALLDONTLIE API, pulled once in October 2026. It was chosen "
        "because its terms allow publishing and modeling the data; salary sites that forbid "
        "scraping or predictive use were avoided.",
    ),
    (
        "Photos and colors",
        "Player headshots load from NBA.com's image server. Team stripes use each club's "
        "official colors.",
    ),
]

HOW_IT_WORKS = f"""
**Compare players.** Two players' game-by-game points, rebounds, and assists, then their
advanced rates from NBA.com: true shooting (TS%), usage (USG%), pace, and PIE, the league's
share-of-game-events metric. Per-36-minute production is computed from the game logs.

**Contract value.** A Ridge regression learns what the market pays for (minutes, scoring,
rebounding, assists, usage, true shooting, PIE, and age) from market-priced contracts and max
deals. Each player is valued by a model that never saw his own salary, and a gap inside the
model's typical miss counts as fairly paid. Rookie-scale, minimum-level, and partial-season
deals aren't market prices, so they're valued but never judged.

**Insights.** Four tests on one player-season:
- *Absence & rest:* how a player performs in the games right after missing time, compared with
  his own season average. It measures games missed, never why.
- *Performance vs. defense:* per-game stats against the league's better and worse defenses,
  split at the median defensive rating.
- *Hot hand:* a permutation test. Each game's shots are shuffled thousands of times to see how
  often chance alone looks as streaky as what really happened.
- *Shot chart:* field-goal percentage by court zone. Zones with fewer than
  {shot_chart.DEFAULT_MIN_ATTEMPTS} attempts are muted rather than shown as 0% or 100%.

**Predictions.** Two forecasts for one player-season:
- *Next-game points:* a Ridge regression on trailing averages, rest, and home/away, trained on
  one full season and scored on the next one, which it never saw.
- *Season forecast:* exponential smoothing on the player's own games, where every forecast uses
  only the games before it.
"""

GROUND_RULES = """
- **Every model sits beside a simple baseline.** A forecast only means something next to the
  obvious guess, like a player's recent average, so each model is shown with the baseline it
  has to beat, including when it doesn't beat it.
- **Only unseen data counts.** Models are scored on games or contracts they never trained on,
  and time-ordered checks never let a forecast see the future.
- **Ties are called ties.** When a model's edge is inside its confidence interval, the app says
  it's tied instead of claiming a win.
- **Small samples get a flag, not a number.** Too few shots, games, or absences returns "not
  enough data" instead of a misleading percentage.
- **No made-up data.** If live data fails, the app shows a labeled snapshot of real data or an
  error.
- **Limits are stated up front.** No free injury feed exists, defensive-scheme data is
  proprietary, and box scores undervalue defense. Each page says what it can't see.
"""

DESIGN = """
The look borrows from the court: a maple floor, chalk-white lines, and navy arena seats. On
Compare Players, each player is drawn in his team's color, on his card, in every chart, and in
the contract plot. Dark team colors are lightened until they stand out on the charts, and when
two players' colors are too close to tell apart (teammates, say), Player 2 switches to his
team's second color. Team colors weren't designed with color blindness in mind, so every
tooltip also names the player. The other charts use a colorblind-safe palette.
"""

BUILT_WITH = f"""
Python and Streamlit, with pandas, scikit-learn (training only), statsmodels, SciPy, and Altair.
Over 200 unit tests run on every push with GitHub Actions. The most important ones guard the
evaluation itself, for example that no model is ever scored on data it trained on.

Built by Aryan Patel. The code and the full write-up, with every confidence interval and
cross-validation detail, are [on GitHub]({REPO_URL}).
"""


def _tie_or_win(comparison: dict) -> str:
    if comparison["ci_low"] > 0:
        return "Model is better"
    if comparison["ci_high"] < 0:
        return "Baseline is better"
    return "Tied: the gap is inside the 95% confidence interval"


def results_table(model: dict, contract_summary: dict, league_forecast: dict) -> pd.DataFrame:
    """One row per question: the method's typical miss next to its simple baseline's."""
    holdout = model["holdout_metrics"]
    contract_metrics = contract_summary["cv_metrics"]
    shipped = contract_metrics[contract_summary["shipped_model"]]
    baseline = contract_metrics["baseline_minutes_only"]
    rows = [
        {
            "Question": "Next-game points",
            "Method": "Ridge regression",
            "Typical miss": f"{holdout['ridge']['mae']:.2f} pts",
            "Baseline": "10-game average",
            "Baseline's miss": f"{holdout['baseline_roll10']['mae']:.2f} pts",
            "Takeaway": _tie_or_win(model["holdout_comparisons"]["baseline_roll10"]),
        },
        {
            "Question": "Season forecast",
            "Method": "Exponential smoothing",
            "Typical miss": f"{league_forecast['ses_mae_pts']:.2f} pts",
            "Baseline": "Season-to-date average",
            "Baseline's miss": f"{league_forecast['season_mean_mae_pts']:.2f} pts",
            "Takeaway": (
                f"Tied: smoothing does better for only {league_forecast['ses_beats_season_mean_pts']:.0%} "
                "of players"
            ),
        },
        {
            "Question": "Contract value",
            "Method": contract_value.MODEL_LABELS[contract_summary["shipped_model"]].capitalize(),
            "Typical miss": f"{shipped['typical_ratio']:.2f}x",
            "Baseline": "Minutes-only model",
            "Baseline's miss": f"{baseline['typical_ratio']:.2f}x",
            "Takeaway": (
                f"Slightly better, but production explains only {shipped['r2_log']:.0%} of "
                "salary differences"
            ),
        },
    ]
    return pd.DataFrame(rows)


def render() -> None:
    st.html(
        ui.page_header(
            "About",
            "About this app",
            "What it does, where every number comes from, and why it's built the way it is.",
        )
    )

    with st.container(key="about"):
        st.html(ui.section_header("What it is"))
        st.markdown(
            "An NBA analytics app built as a data-science portfolio project. It compares players, "
            "tests popular basketball beliefs (is there a hot hand? do players come back rusty?), "
            "forecasts next-game stats, and asks whether a contract matches a player's production. "
            "The goal isn't the flashiest prediction. It's to show how well each question can "
            "actually be answered from public data, and to say so when the answer is \"not much.\""
        )

        st.html(ui.section_header("Where the numbers come from"))
        for col, (title, body) in zip(st.columns(len(SOURCES), gap="medium"), SOURCES):
            col.html(ui.tile(title, body))

        st.html(ui.section_header("How each page works"))
        st.markdown(HOW_IT_WORKS)

        st.html(ui.section_header("Why it's built this way", "Ground rules"))
        st.markdown(GROUND_RULES)

        st.html(ui.section_header("Results", "Scored on data each model never saw"))
        try:
            model = modeling.load_linear_model()
        except FileNotFoundError:
            model = None
        contract_summary = contract_value.load_model_summary()
        if model is None or contract_summary is None:
            st.info("Model results haven't been built yet; see the README for the training scripts.")
        else:
            st.table(results_table(model, contract_summary, forecast.LEAGUE_RESULT).set_index("Question"))
            st.caption(
                "Typical miss is the mean absolute error. For salaries it's a ratio: 1.52x means the "
                "estimate is typically about 52% too high or too low. Single-game scoring is mostly "
                "noise around a player's level, which is why every method lands close together."
            )

        st.html(ui.section_header("Design"))
        st.markdown(DESIGN)

        st.html(ui.section_header("Built with"))
        st.markdown(BUILT_WITH)
