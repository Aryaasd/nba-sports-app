# NBA Player Stats App

An interactive Streamlit app for comparing NBA player stats and exploring hypothesis-driven data-science insights, powered by `nba_api`.

[![CI](https://github.com/Aryaasd/nba-sports-app/actions/workflows/ci.yml/badge.svg)](https://github.com/Aryaasd/nba-sports-app/actions/workflows/ci.yml)

---

## Live Demo

**[Try it live](https://nba-sports-app-jneksyc8sa3dsvprpkbxdq.streamlit.app/)**

> **A known limitation, not a bug:** `stats.nba.com` throttles or times out requests from cloud/datacenter IP ranges (Streamlit Community Cloud, Heroku, Render, etc.) far more aggressively than from a residential connection. The deployed demo may occasionally show a connectivity error instead of live data as a result — a widely-documented constraint of building on `nba_api` from free cloud hosting, confirmed here after adding retries with backoff didn't clear it. The GIF below was recorded locally, where this restriction doesn't apply; running the app locally (see below) always has full access.

![demo](docs/demo.gif)

---

## Key Features

* **Compare Two Players:** Season game logs for any two players, side-by-side or overlaid, PTS/REB/AST charted by game.
* **Advanced Metrics:** TS%, USG%, PACE, PIE (pulled directly from `nba_api`'s advanced stats), plus hand-computed PTS/REB/AST per-36 rates.
* **Insights tab — four hypothesis-driven analyses for one player-season, not just dashboard metrics:**
  * **Absence & Rest Impact** — how a player performs in the games right after missing time, vs. their own season baseline.
  * **Performance vs. Defense** — scoring split against opponents bucketed by measured `DEF_RATING`.
  * **Hot Hand Fallacy** — an actual permutation test of whether a player's makes/misses are streakier than chance, in the spirit of Gilovich, Vallone & Tversky (1985) and the Miller & Sanjurjo (2015) bias correction.
  * **Shot Chart** — a hand-drawn court with a field-goal-percentage heatmap, muted for low-attempt bins so small samples don't look like real signal.
* **Cached:** Every `nba_api` call is wrapped in `@st.cache_data`, so switching chart views or re-selecting a player never re-hits the network unnecessarily.

Every Insights analysis is deliberately honest about what it can't show — see [How It Works](#how-it-works) for the specific data limitations each one works around.

---

## Tech Stack

* **Core:** Python, Streamlit
* **Data:** Pandas, NumPy, `nba_api`
* **Visualization:** Altair
* **Testing/CI:** pytest, ruff, GitHub Actions

---

## Getting Started

### Prerequisites

* Python 3.9+
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
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run sports_app.py
   ```

Your app should now be open and running in your default web browser.

---

## How It Works

The app is split into small, single-purpose modules instead of one script:

* **`data.py`** — every `nba_api` call, each wrapped in `@st.cache_data(ttl=3600)`. Advanced stats and team stats are fetched once per season and shared across every player, not re-fetched per selection.
* **`metrics.py`** — pure logic (season strings, player lookup, TS%/per-36 math). No `streamlit` or `nba_api` imports, so it's fast to unit test in isolation.
* **`charts.py`** — Altair chart builders, plus a registered dark theme (validated CVD-safe categorical and diverging color pairs) shared by every chart in the app.
* **`insights/`** — the four Insights-tab analyses, each in its own module (`absence.py`, `defense_tiers.py`, `hot_hand.py`, `shot_chart.py`), each honest about a real data limitation rather than overclaiming what free NBA data can show:
  * No free live injury-designation feed exists anywhere, so **Absence & Rest Impact** measures games missed and return-game performance, never *why* a game was missed.
  * True defensive-scheme data (zone vs. man, coverage type) is proprietary tracking data, so **Performance vs. Defense** compares against opponents bucketed by the league's own measured `DEF_RATING` instead.
  * **Hot Hand Fallacy** runs a permutation test (shuffling each game's own shot sequence thousands of times) rather than a naive streak count, specifically to sidestep the selection bias in the original 1985 hot-hand analysis.
  * Every analysis returns an explicit "not enough data" result instead of a misleading number when the sample is too small — a single absence event, or a shot chart with only a handful of attempts, says so rather than guessing.
* **`sports_app.py`** — thin Streamlit UI glue. It contains no data-fetching or stat-math logic of its own.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

All pure-logic modules (`metrics.py`, `charts.py`, `data.py`, and everything in `insights/`) are unit tested with mocked `nba_api` calls — tests never hit the live network, so they're fast and don't flake on rate limits. The Streamlit UI itself isn't tested (that would need `streamlit.testing.v1.AppTest`); it's kept thin enough that the logic it calls is what's actually verified.

---

## Deploying Your Own Copy

1. Push your fork to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app**, select your repo/branch, and set the entry file to `sports_app.py`.
4. Deploy.

---

## Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
