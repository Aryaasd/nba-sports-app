# NBA Player Stats App

An interactive Streamlit app for comparing NBA players, testing hypotheses about their performance, and forecasting their next games. It runs on `nba_api` data, and every model is reported next to the simple baseline it has to beat.

[![CI](https://github.com/Aryaasd/nba-sports-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Aryaasd/nba-sports-app/actions/workflows/ci.yml)

---

## Live Demo

**[Try it live](https://nba-sports-app-jneksyc8sa3dsvprpkbxdq.streamlit.app/)**

> **About the hosted demo:** `stats.nba.com` blocks requests from most cloud hosts, including Streamlit Community Cloud. When live data can't be reached, the app offers a one-click switch to a bundled **real** snapshot (LeBron James and Stephen Curry, 2025-26), and a banner says so. Any other selection shows an error rather than made-up data. Run it locally (see below) for live data on every player.

![demo](docs/demo.gif)

---

## Key Features

* **Compare Two Players:** season game logs for any two players, side by side or overlaid.
* **Advanced Metrics:** TS%, USG%, PACE, and PIE from `nba_api`'s advanced stats, plus hand-computed per-36 rates.
* **Insights:** four hypothesis-driven analyses for one player-season:
  * **Absence & Rest Impact** — performance in the games right after missed time, compared with the player's own season baseline.
  * **Performance vs. Defense** — scoring against opponents bucketed by measured `DEF_RATING`.
  * **Hot Hand Fallacy** — a permutation test of whether makes and misses are streakier than chance, in the spirit of Gilovich, Vallone & Tversky (1985) and the Miller & Sanjurjo (2015) bias correction.
  * **Shot Chart** — a field-goal-percentage heatmap on a hand-drawn court, with low-attempt bins muted so small samples don't look like signal.
* **Predictions:** two forecasting techniques, each scored only on games it hadn't seen:
  * **Next-Game Points Model** — a Ridge regression trained on 20,932 player-games. It shows a per-player backtest chart and a next-game projection.
  * **Season Forecast** — simple exponential smoothing on one player's own season, validated walk-forward. The fitted smoothing weight is itself a finding (see below).
* **Resilient demo:** a circuit breaker and a labeled sample-data fallback, so the hosted app never dead-ends on a blocked API.

---

## Model Evaluation

The goal was to find out honestly how well next-game stats *can* be predicted, not to make a model look good. Two rules apply to every number below: a prediction only ever uses games played before it, and every model is compared against simple averages.

### Next-game points: Ridge regression

**Setup**
- **Training data:** every player's 2024-25 regular season (20,932 player-games after each player's first 10).
- **Features:** trailing 5- and 10-game averages of points, rebounds, assists, and minutes, plus days of rest and home/away.
- **Validation:** 5-fold time-series cross-validation, split on whole dates so a single night never straddles train and validation. The **entire 2025-26 season** is a holdout the model never saw.

| Model | CV MAE (2024-25) | Holdout MAE (2025-26) | Holdout RMSE |
|---|---:|---:|---:|
| 5-game average (naive baseline) | 4.914 | 4.867 | 6.376 |
| 10-game average (stronger baseline) | 4.777 | 4.716 | 6.181 |
| **Ridge regression** | **4.759** | **4.708** | **6.115** |
| Random forest (comparison only) | 4.779 | 4.723 | 6.132 |

**What this shows**
- The model beats a 5-game average by 3.3%, but a plain 10-game average by only **0.2%**. A longer averaging window captures about 95% of the gain; the other features add little.
- Single-game scoring is mostly noise around a player's level. With a typical miss near 4.7 points, the floor is close to what any box-score model can reach.
- A nonlinear random forest didn't beat Ridge, so the simpler, interpretable model ships.

### Season forecast: exponential smoothing

Each player's season is treated as a time series. At every game *k*, three forecasters see only games 1..*k*−1 and predict game *k* (rolling-origin evaluation). Results across **367 players with 40+ games in 2025-26**:

| Stat | Smoothing MAE | Season-mean MAE | Last-game MAE | Smoothing beats season mean | Median fitted α |
|---|---:|---:|---:|---:|---:|
| PTS | 4.72 | 4.74 | 6.05 | 39% of players | 0.00 |
| REB | 1.95 | 1.93 | 2.49 | 37% of players | 0.00 |
| AST | 1.38 | 1.37 | 1.72 | 39% of players | 0.00 |

- Smoothing ties the season-to-date average, and both beat "last game carried forward" by about 22%.
- **The median fitted smoothing weight is 0**: for the typical player, the best forecast puts no extra weight on recent games. That's the hot-hand question asked per game instead of per shot, with the same answer — recent form carries little signal beyond a player's season level.

### Limitations

* Box scores only. There's no injury, minutes-restriction, opponent, or lineup information; no free source exists for the first two.
* The model is trained on one season. A different era, or a rule change, could shift the relationships.
* The next-game projection needs rest days and home/away as user inputs, because the app has no schedule feed.
* Point forecasts only; no prediction intervals.

Reproduce the numbers with `python -m scripts.train_model` (about 10s) and `python -m scripts.evaluate_forecast` (about 1 min). Both pull live data, so run them from a non-cloud machine.

---

## Tech Stack

* **Core:** Python, Streamlit
* **Data:** Pandas, NumPy, `nba_api`, Parquet (bundled snapshot)
* **Modeling:** scikit-learn (Ridge, random forest, `TimeSeriesSplit`; training only), statsmodels (exponential smoothing)
* **Visualization:** Altair
* **Testing/CI:** pytest, ruff, GitHub Actions

---

## Getting Started

### Prerequisites

* Python 3.11+
* `pip`

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aryaasd/nba-sports-app.git
   cd nba-sports-app
   ```

2. **(Recommended) Create and activate a virtual environment:**
   * **Windows:**
     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run sports_app.py
   ```

---

## How It Works

The app is split into small, single-purpose modules:

* **`data.py`** — every `nba_api` call. Each is cached for an hour with `@st.cache_data`, retried once, and guarded by a circuit breaker: after a network failure, live calls pause for 10 minutes instead of re-waiting out the timeout on every request. On failure it falls back to the bundled snapshot, but only for the exact selections that snapshot covers.
* **`metrics.py`** — pure logic: season strings, player lookup, TS% and per-36 math.
* **`modeling.py`** — next-game feature engineering and inference, shared by training and the live app so the two can't drift apart.
  * Each feature is built from the player's own earlier games only: shifted one game, computed per player, and requiring a full window.
  * The model ships as plain JSON coefficients (`model/next_game_points.json`), so the deployed app doesn't need scikit-learn.
* **`model_training.py`** — date-block time-series CV, model fitting, and coefficient export (scikit-learn).
* **`insights/`** — one module per analysis (`absence.py`, `defense_tiers.py`, `hot_hand.py`, `shot_chart.py`, `forecast.py`). Each is honest about a real data limitation:
  * No free injury-designation feed exists, so **Absence & Rest Impact** measures games missed, never *why* they were missed.
  * Defensive-scheme data is proprietary, so **Performance vs. Defense** buckets opponents by measured `DEF_RATING` instead.
  * **Hot Hand** shuffles each game's own shot sequence thousands of times, to sidestep the selection bias in the original 1985 analysis.
  * Every analysis returns an explicit "not enough data" result instead of a misleading number from a tiny sample.
* **`sample_data_loader.py` + `sample_data/`** — the bundled real snapshot used when the live API is unreachable. See [`sample_data/README.md`](sample_data/README.md) for its provenance.
* **`scripts/`** — one-off jobs:
  * `train_model` — trains and evaluates the model, and writes the JSON.
  * `evaluate_forecast` — the league-wide forecast table above.
  * `capture_sample_data` — refreshes the snapshot.
* **`charts.py`** — Altair chart builders and a dark theme with colorblind-safe categorical and diverging colors.
* **`sports_app.py`** — thin Streamlit UI glue, with no data-fetching or stat-math logic of its own.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

Every logic module is unit tested with mocked `nba_api` calls, so tests never hit the network. The tests that matter most guard the evaluation itself:
* CV folds never train on a date at or after their validation dates.
* Walk-forward forecasts never see the game they predict.
* Rolling features never include the current game or another player's history.
* The exported JSON coefficients reproduce scikit-learn's predictions.
* The fallback never serves sample data for a selection outside the snapshot.

The Streamlit UI itself isn't tested; it's kept thin enough that the logic it calls is what's verified.

---

## Deploying Your Own Copy

1. Push your fork to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app**, select your repo/branch, and set the entry file to `sports_app.py`.
4. Deploy.

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
