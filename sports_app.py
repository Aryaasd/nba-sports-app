import pandas as pd
import streamlit as st

import charts
import data
import metrics
import modeling
import sample_data_loader
from insights import absence, defense_tiers, forecast, hot_hand, shot_chart

st.set_page_config(page_title="NBA Sports App", layout="wide")

st.title("NBA Sports App")

page = st.sidebar.radio("Navigate", ["Home", "Compare Players", "Insights", "Predictions"])

SAMPLE_LABEL = (
    f"{' / '.join(sample_data_loader.SAMPLE_PLAYERS.values())}, {sample_data_loader.SAMPLE_SEASON}"
)
# 10 games of history before the first prediction, plus enough predictions to judge.
MIN_PREDICTION_GAMES = 15


@st.cache_data(ttl=data.CACHE_TTL_SECONDS)
def _cached_hot_hand_summary(shot_df: pd.DataFrame, n_permutations: int = 2000, seed: int = 42) -> dict:
    # The permutation test itself is pure computation (no nba_api call), but at
    # ~1s for 2000 shuffles it's worth caching same as a network fetch -- Streamlit
    # reruns this whole script on every widget interaction on the Insights page.
    return hot_hand.summarize_hot_hand(shot_df, n_permutations=n_permutations, seed=seed)


@st.cache_data(ttl=data.CACHE_TTL_SECONDS)
def _cached_forecast_summary(game_log_df: pd.DataFrame, stat_col: str) -> dict:
    return forecast.summarize_forecast(game_log_df, stat_col)


def _sidebar_select(label: str, options: list[str], key: str, default: str) -> str:
    # Default via session_state, not `index=`: Streamlit warns when a widget has a
    # non-default index *and* a callback (the sample-data button) writes its key.
    if key not in st.session_state:
        st.session_state[key] = options[metrics.safe_selectbox_index(options, default)]
    return st.sidebar.selectbox(label, options, key=key)


def _select_sample_data(season_key: str, player_keys: list[str]) -> None:
    # Must run as an on_click callback: writing a widget's session_state after
    # the widget has rendered in the same run raises StreamlitAPIException.
    st.session_state[season_key] = sample_data_loader.SAMPLE_SEASON
    for key, name in zip(player_keys, sample_data_loader.SAMPLE_PLAYERS.values()):
        st.session_state[key] = name


def _stop_with_fetch_error(exc: Exception, season_key: str, player_keys: list[str]) -> None:
    st.error(
        "Couldn't load live NBA data for this selection -- the NBA stats API didn't "
        "respond. It blocks many cloud servers, including the one this demo is hosted on."
    )
    st.button(
        f"Explore with cached sample data ({SAMPLE_LABEL})",
        on_click=_select_sample_data,
        args=(season_key, player_keys),
        type="primary",
    )
    with st.expander("Error details"):
        st.code(str(exc), language=None)
    st.stop()


def _show_sample_banner(*used_sample_flags: bool) -> None:
    # Worded to stay true when only a league-wide table fell back for a player whose own
    # game log loaded live -- the snapshot's league tables cover every player.
    if any(used_sample_flags):
        players = " and ".join(sample_data_loader.SAMPLE_PLAYERS.values())
        st.warning(
            f"Some of this page comes from a cached {sample_data_loader.SAMPLE_SEASON} snapshot of "
            "real data -- the live NBA stats API is unreachable from this host right now. The "
            f"snapshot covers {players} plus league-wide tables; any other player shows an error "
            "rather than made-up data."
        )


try:
    all_players = data.get_all_players()
except data.PlayerStatsFetchError as exc:
    st.error(str(exc))
    st.stop()

player_index = metrics.build_player_index(all_players)
player_names = metrics.get_player_names(all_players)
seasons = metrics.get_recent_seasons()
default_season = metrics.get_default_season()

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "A stats app for comparing NBA players and testing real hypotheses about "
        "their performance, built on live data from `nba_api`. Pick a section from "
        "the sidebar to get started."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Compare Players**")
        st.write(
            "Season game logs for any two players, charted side by side or "
            "overlaid. PTS, REB, and AST by game, sorted and cached so switching "
            "views is instant."
        )
    with col2:
        st.markdown("**Advanced Metrics**")
        st.write(
            "TS%, USG%, PACE, and PIE pulled directly from `nba_api`'s advanced "
            "stats, plus hand-computed PTS/REB/AST per-36 rates for the same two "
            "players."
        )
    with col3:
        st.markdown("**Insights**")
        st.write(
            "Four hypothesis-driven analyses for one player-season: absence and "
            "rest impact, performance against elite vs. weak defenses, a real "
            "permutation test for the hot-hand fallacy, and a shot-chart "
            "efficiency heatmap."
        )
    with col4:
        st.markdown("**Predictions**")
        st.write(
            "A next-game points model trained on 20,000+ player-games, and a "
            "season forecast using exponential smoothing. Both are scored only on "
            "games they hadn't seen, against simple averages."
        )

    st.divider()
    st.caption(
        "Every analysis states what it can't show rather than overclaiming: no free "
        "live injury feed exists, true defensive-scheme data is proprietary, small "
        "samples are flagged instead of guessed at, and each model is reported next "
        "to the simple baseline it has to beat."
    )

elif page == "Compare Players":
    st.subheader("Compare Two Players")

    season = _sidebar_select("Season", seasons, "compare_season", default_season)

    player1_name = _sidebar_select("Player 1", player_names, "compare_player1", "LeBron James")
    player2_name = _sidebar_select("Player 2", player_names, "compare_player2", "Kevin Durant")

    player1_id = metrics.get_player_id(player_index, player1_name)
    player2_id = metrics.get_player_id(player_index, player2_name)

    try:
        df1, sample1 = data.fetch_player_game_log(player1_id, season)
        df2, sample2 = data.fetch_player_game_log(player2_id, season)
        advanced_df, sample_advanced = data.fetch_advanced_stats(season)
    except data.PlayerStatsFetchError as exc:
        _stop_with_fetch_error(exc, "compare_season", ["compare_player1", "compare_player2"])

    _show_sample_banner(sample1, sample2, sample_advanced)

    if df1.empty or df2.empty:
        st.error("Could not fetch stats for one or both players this season.")
    else:
        view = st.radio("Chart View", ["Side by Side", "Overlay"], horizontal=True)

        if view == "Side by Side":
            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(charts.prepare_line_chart_data(df1))
                st.caption(player1_name)
            with col2:
                st.line_chart(charts.prepare_line_chart_data(df2))
                st.caption(player2_name)
        else:
            combined_df = pd.concat(
                [
                    charts.melt_for_overlay(df1, player1_name),
                    charts.melt_for_overlay(df2, player2_name),
                ]
            )
            st.altair_chart(charts.build_overlay_chart(combined_df), use_container_width=True, theme=None)

        st.subheader("Advanced Metrics")
        row1 = metrics.extract_player_advanced_row(advanced_df, player1_id)
        row2 = metrics.extract_player_advanced_row(advanced_df, player2_id)
        player_stats = {
            player1_name: {
                **metrics.select_advanced_columns(row1),
                **metrics.season_per36_totals(df1),
            },
            player2_name: {
                **metrics.select_advanced_columns(row2),
                **metrics.season_per36_totals(df2),
            },
        }
        table_col, note_col = st.columns([2, 1])
        with table_col:
            st.dataframe(metrics.build_comparison_table(player_stats), use_container_width=True)
        with note_col:
            st.markdown("**Reading this table**")
            st.caption(
                "TS% and USG% come straight from `nba_api`'s Advanced measure type. "
                "PACE is the team's estimated possessions per 48 minutes while the "
                "player is on the floor. PIE is the league's own share-of-game-events "
                "metric. Per-36 rates are computed from this season's game log, not "
                "a separate API call."
            )

elif page == "Insights":
    st.subheader("Insights")
    st.caption(
        "Four small, hypothesis-driven analyses for one player-season -- honest "
        "about what the data can and can't show, not a prediction."
    )

    season = _sidebar_select("Season", seasons, "insights_season", default_season)
    player_name = _sidebar_select("Player", player_names, "insights_player", "LeBron James")
    player_id = metrics.get_player_id(player_index, player_name)

    try:
        game_log_df, sample_log = data.fetch_player_game_log(player_id, season)
        all_teams = data.get_all_teams()
        team_advanced_df, sample_team_stats = data.fetch_team_advanced_stats(season)
        shot_df, sample_shots = data.fetch_shot_chart(player_id, season)
    except data.PlayerStatsFetchError as exc:
        _stop_with_fetch_error(exc, "insights_season", ["insights_player"])

    if game_log_df.empty:
        st.error(f"No {season} game data for {player_name}.")
        st.stop()

    # PlayerGameLog has no TEAM_ID column -- derive it from the player's own
    # MATCHUP abbreviation instead (first token, before "vs."/"@").
    own_abbr = defense_tiers.extract_own_team_abbreviation(game_log_df.iloc[0]["MATCHUP"])
    team_id = metrics.build_abbreviation_to_team_id_index(all_teams).get(own_abbr)
    team_game_dates, sample_dates, team_dates_error = [], False, None
    if team_id:
        try:
            team_game_dates, sample_dates = data.fetch_team_game_dates(team_id, season)
        except data.PlayerStatsFetchError as exc:
            team_dates_error = str(exc)

    _show_sample_banner(sample_log, sample_team_stats, sample_shots, sample_dates)

    tab_absence, tab_defense, tab_hot_hand, tab_shot_chart = st.tabs(
        ["Absence & Rest Impact", "Performance vs. Defense", "Hot Hand Fallacy", "Shot Chart"]
    )

    with tab_absence:
        st.write(
            "No free live injury-designation feed exists anywhere, so this measures "
            "**games missed and return-game performance** -- never *why* a game was missed."
        )
        if team_dates_error:
            st.error(team_dates_error)

        missed_dates = absence.find_missed_games(team_game_dates, game_log_df["GAME_DATE"].tolist())
        absence_gaps = absence.identify_absence_gaps(team_game_dates, missed_dates)
        result = absence.compute_return_game_impact(game_log_df, absence_gaps)

        if result["insufficient_data"]:
            st.info(
                f"Not enough absence events for {player_name} this season "
                f"({result['n_absence_events']} found, need at least {absence.MIN_ABSENCE_EVENTS}) "
                "to say anything meaningful."
            )
        else:
            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.write(
                    f"**{result['n_absence_events']} absence events**, "
                    f"**{result['n_return_games']} return games** analyzed."
                )
                st.altair_chart(
                    charts.build_delta_bar_chart(result["stats"]), use_container_width=True, theme=None
                )
            with table_col:
                st.markdown("**Return-game vs. season baseline**")
                deltas_df = pd.DataFrame(result["stats"]).T
                st.dataframe(deltas_df.style.format("{:.2f}"), use_container_width=True)
                st.caption(
                    "Assumes the player stayed on one team all season -- a mid-season "
                    "trade will misregister old-team games as \"missed.\""
                )

    with tab_defense:
        st.write(
            "True defensive-scheme data (zone vs. man, coverage type) is proprietary "
            "tracking data not published anywhere for free. This instead compares "
            "performance against opponents bucketed by their measured **DEF_RATING**."
        )
        team_id_to_abbr = metrics.build_team_abbreviation_index(all_teams)
        tiers = defense_tiers.bucket_teams_by_defense(team_advanced_df, team_id_to_abbr)
        tier_result = defense_tiers.compare_performance_by_tier(game_log_df, tiers)

        if tier_result.empty:
            st.info(f"Not enough matchup data to bucket {player_name}'s opponents this season.")
        else:
            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.altair_chart(
                    charts.build_defense_tier_chart(tier_result), use_container_width=True, theme=None
                )
            with table_col:
                st.markdown("**Per-game averages by opponent tier**")
                st.dataframe(
                    tier_result.style.format({"PTS": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "Tiers are a median DEF_RATING split across the league -- "
                    "\"Games\" shows how many of this season's matchups landed in "
                    "each tier, so the split's sample size is never hidden."
                )

    with tab_hot_hand:
        st.write(
            "A real hypothesis test, not a dashboard metric. Gilovich, Vallone & "
            "Tversky (1985) argued the \"hot hand\" is a cognitive illusion; "
            "Miller & Sanjurjo (2015) later showed the original test itself was "
            "subtly biased. This uses a permutation test to sidestep that bias: "
            "shuffle each game's own shot sequence thousands of times and see how "
            "often shuffled data looks at least as streaky as reality did."
        )
        hot_hand_result = _cached_hot_hand_summary(shot_df)

        if hot_hand_result["insufficient_data"]:
            st.info(
                f"Not enough shot data for {player_name} this season "
                f"({hot_hand_result['total_shots']} shots) to run the test."
            )
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("P(make | prev make)", f"{hot_hand_result['p_make_after_make']:.1%}")
            col2.metric("P(make | prev miss)", f"{hot_hand_result['p_make_after_miss']:.1%}")
            col3.metric("p-value", f"{hot_hand_result['p_value']:.3f}")

            chart_col, note_col = st.columns([3, 2])
            with chart_col:
                st.altair_chart(
                    charts.build_null_distribution_chart(
                        hot_hand_result["null_distribution"], hot_hand_result["observed_diff"]
                    ),
                    use_container_width=True,
                    theme=None,
                )
            with note_col:
                st.markdown("**Reading this test**")
                st.write(
                    f"If there were truly no streakiness, a gap this large would show "
                    f"up in shuffled data about **{hot_hand_result['p_value']:.1%}** of "
                    "the time -- that's the p-value. The blue histogram is the null "
                    "distribution built from {n} shuffles of {player}'s own shot "
                    "sequence; the red line marks where reality actually fell.".format(
                        n=hot_hand_result["n_permutations"], player=player_name
                    )
                )
                st.caption(
                    f"Built from {hot_hand_result['total_shots']} shots this season "
                    f"({hot_hand_result['n_after_make']} following a make, "
                    f"{hot_hand_result['n_after_miss']} following a miss)."
                )

    with tab_shot_chart:
        st.write(
            "Field-goal percentage by ~4-foot spatial bin. Bins under 3 attempts "
            "render muted grey instead of a misleading 100%/0% from a single shot."
        )
        if shot_df.empty:
            st.info(f"No shot data for {player_name} this season.")
        else:
            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.altair_chart(shot_chart.build_shot_chart(shot_df), use_container_width=False, theme=None)
            with table_col:
                st.markdown("**Best zones this season**")
                binned = shot_chart.bin_shots(shot_df)
                qualified = binned[binned["attempts"] >= shot_chart.DEFAULT_MIN_ATTEMPTS].copy()
                if qualified.empty:
                    st.caption("No zone has enough attempts yet to rank.")
                else:
                    qualified["Zone"] = [
                        shot_chart.label_zone(row.x0, row.y0) for row in qualified.itertuples()
                    ]
                    top_zones = (
                        qualified.sort_values("fg_pct", ascending=False)
                        .head(5)[["Zone", "attempts", "makes", "fg_pct"]]
                        .rename(columns={"attempts": "Attempts", "makes": "Makes", "fg_pct": "FG%"})
                    )
                    st.dataframe(
                        top_zones.style.format({"FG%": "{:.1%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Zone labels are a distance-from-hoop approximation, not the "
                        "official zone geometry -- a supplementary summary, not a "
                        "second source of truth."
                    )

elif page == "Predictions":
    st.subheader("Predictions")
    st.caption(
        "Two forecasting techniques for one player-season. Each is scored only on games it "
        "hadn't seen yet, next to the simple averages it has to beat."
    )

    season = _sidebar_select("Season", seasons, "predictions_season", default_season)
    player_name = _sidebar_select("Player", player_names, "predictions_player", "LeBron James")
    player_id = metrics.get_player_id(player_index, player_name)

    try:
        game_log_df, sample_log = data.fetch_player_game_log(player_id, season)
    except data.PlayerStatsFetchError as exc:
        _stop_with_fetch_error(exc, "predictions_season", ["predictions_player"])

    _show_sample_banner(sample_log)

    if game_log_df.empty:
        st.error(f"No {season} game data for {player_name}.")
        st.stop()

    tab_model, tab_forecast = st.tabs(["Next-Game Points Model", "Season Forecast"])

    with tab_model:
        try:
            model = modeling.load_linear_model()  # a ~2 KB JSON; not worth caching
        except FileNotFoundError:
            model = None
        if model is None:
            st.info(
                "No trained model found. Run `python -m scripts.train_model` to create "
                "`model/next_game_points.json`."
            )
        else:
            holdout = model["holdout_metrics"]
            ridge_mae = holdout["ridge"]["mae"]
            roll5_mae = holdout["baseline_roll5"]["mae"]
            roll10_mae = holdout["baseline_roll10"]["mae"]
            st.write(
                f"A Ridge regression trained on every player's {model['training_season']} regular "
                f"season ({model['n_train_rows']:,} player-games) predicts next-game points from "
                "trailing 5- and 10-game averages of points, rebounds, assists and minutes, plus "
                f"rest days and home/away. Scored on the entire {model['holdout_season']} season, "
                "which it never saw:"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Model: typical miss", f"{ridge_mae:.2f} pts")
            baselines = [(col2, "10-game average", roll10_mae), (col3, "5-game average", roll5_mae)]
            for col, label, mae in baselines:
                col.metric(label, f"{mae:.2f} pts", f"{mae - ridge_mae:+.2f} vs. model", delta_color="off")
            comparisons = model["holdout_comparisons"]
            st.caption(
                f"Typical miss = mean absolute error over {model['n_holdout_rows']:,} unseen "
                "player-games. Versus a 5-game average, the model "
                f"{modeling.describe_comparison(comparisons['baseline_roll5'])}; versus a plain "
                f"10-game average, it {modeling.describe_comparison(comparisons['baseline_roll10'])}. "
                "Intervals come from resampling whole players. Single-game scoring is mostly noise "
                "around a player's level, so box-score features can't get much closer."
            )

            if len(game_log_df) < MIN_PREDICTION_GAMES:
                st.info(
                    f"{player_name} played {len(game_log_df)} games in {season}. The model needs 10 "
                    "earlier games before its first prediction, so there aren't enough predictions "
                    f"here to judge it (need at least {MIN_PREDICTION_GAMES} games)."
                )
            else:
                backtest = modeling.player_backtest(game_log_df.assign(PLAYER_ID=player_id), model)
                series = ["Actual", "Model", "Baseline (10-game avg)"]
                chart_col, side_col = st.columns([3, 2])
                with chart_col:
                    st.markdown(f"**{player_name}, {season}: each prediction, made before the game**")
                    long_df = backtest.melt(
                        id_vars="GAME_DATE", value_vars=series, var_name="Series", value_name="Value"
                    )
                    st.altair_chart(
                        charts.build_backtest_chart(long_df, series, y_title="Points"),
                        use_container_width=True,
                        theme=None,
                    )
                with side_col:
                    st.markdown(f"**{player_name}'s typical miss this season**")
                    player_errors = pd.DataFrame(
                        {
                            "Method": ["Model", "10-game average", "5-game average"],
                            "Typical miss (pts)": [
                                (backtest[col] - backtest["Actual"]).abs().mean()
                                for col in ["Model", "Baseline (10-game avg)", "Baseline (5-game avg)"]
                            ],
                        }
                    )
                    st.dataframe(
                        player_errors.style.format({"Typical miss (pts)": "{:.2f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                    last_game = game_log_df["GAME_DATE"].max()
                    st.markdown(f"**Next game after {last_game:%b %d, %Y}**")
                    rest_col, home_col = st.columns(2)
                    days_rest = rest_col.number_input(
                        "Days of rest", 0, modeling.MAX_DAYS_REST, 1, key="predictions_rest"
                    )
                    location = home_col.radio(
                        "Location", ["Home", "Away"], horizontal=True, key="predictions_home"
                    )
                    next_row = modeling.build_next_game_row(game_log_df, days_rest, location == "Home")
                    st.metric("Projected points", f"{modeling.predict_linear(model, next_row)[0]:.1f}")
                    st.caption(
                        f"Typical miss on unseen games is about {ridge_mae:.1f} pts. Rest and location "
                        "are your assumptions -- the app has no schedule feed."
                    )

                if season == model["training_season"]:
                    st.caption(
                        f"In-sample: the model was trained on {season}, so these per-game misses look "
                        "slightly better than they would on new data."
                    )
                elif season < model["training_season"]:
                    st.caption(
                        f"Not a true forecast: the model was trained on {model['training_season']}, a "
                        f"later season, so it learned from data that didn't exist yet in {season}."
                    )
                st.caption(
                    "Knows only this player's recent box scores, rest, and home/away -- not injuries, "
                    "minutes restrictions, the opponent, or trades."
                )

    with tab_forecast:
        st.write(
            "A different technique: forecast each game from this player's own season so far, "
            "with no features and no other players. **Simple exponential smoothing** "
            "(statsmodels) weights recent games more heavily and learns *how much* from the "
            "data. Evaluated walk-forward: every forecast uses only the games before it."
        )
        stat_col = st.radio("Stat", ["PTS", "REB", "AST"], horizontal=True, key="forecast_stat")
        result = _cached_forecast_summary(game_log_df, stat_col)

        if result["insufficient_data"]:
            st.info(
                f"{player_name} played {result['n_games']} games in {season} -- need at least "
                f"{forecast.MIN_TOTAL_GAMES} to evaluate a forecast."
            )
        else:
            walk_forward = result["metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Smoothing: typical miss", f"{walk_forward['ses']['mae']:.2f} {stat_col}")
            col2.metric("Season-to-date average", f"{walk_forward['season_mean']['mae']:.2f} {stat_col}")
            col3.metric("Last game carried forward", f"{walk_forward['naive']['mae']:.2f} {stat_col}")

            chart_col, note_col = st.columns([3, 2])
            with chart_col:
                series = ["Actual", forecast.CHART_LABELS["ses"], forecast.CHART_LABELS["season_mean"]]
                st.altair_chart(
                    charts.build_backtest_chart(result["chart_df"], series, y_title=stat_col),
                    use_container_width=True,
                    theme=None,
                )
            with note_col:
                st.markdown("**What the fitted weight says**")
                st.write(
                    f"Smoothing weight **α = {result['alpha']:.2f}**. "
                    f"{forecast.describe_alpha(result['alpha'])}"
                )
                st.write(f"**{forecast.compare_to_season_mean(walk_forward)}**")
                st.caption(
                    "At α = 0 smoothing *is* the season average; at α = 1 it's last game carried "
                    "forward. Whether a fitted α earns its keep is the Hot Hand question, asked per "
                    "game instead of per shot."
                )
                st.metric(f"Next-game {stat_col} forecast", f"{result['next_forecast']:.1f}")
                st.caption(
                    f"Scored on {result['n_games'] - forecast.MIN_TRAIN_SIZE} walk-forward forecasts; "
                    f"the first {forecast.MIN_TRAIN_SIZE} games are training-only."
                )
