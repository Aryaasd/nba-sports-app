import pandas as pd
import streamlit as st

import charts
import data
import metrics
from insights import absence, defense_tiers, hot_hand, shot_chart

st.set_page_config(page_title="NBA Sports App", layout="wide")

st.title("NBA Sports App")

page = st.sidebar.radio("Navigate", ["Home", "Compare Players", "Insights"])


@st.cache_data(ttl=data.CACHE_TTL_SECONDS)
def _cached_hot_hand_summary(shot_df: pd.DataFrame, n_permutations: int = 2000, seed: int = 42) -> dict:
    # The permutation test itself is pure computation (no nba_api call), but at
    # ~1s for 2000 shuffles it's worth caching same as a network fetch -- Streamlit
    # reruns this whole script on every widget interaction on the Insights page.
    return hot_hand.summarize_hot_hand(shot_df, n_permutations=n_permutations, seed=seed)


try:
    all_players = data.get_all_players()
except data.PlayerStatsFetchError as exc:
    st.error(str(exc))
    st.stop()

player_index = metrics.build_player_index(all_players)
player_names = metrics.get_player_names(all_players)

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "A stats app for comparing NBA players and testing real hypotheses about "
        "their performance, built on live data from `nba_api`. Pick a section from "
        "the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)
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

    st.divider()
    st.caption(
        "Every Insights analysis states what it can't show rather than "
        "overclaiming: no free live injury feed exists, true defensive-scheme "
        "data is proprietary, and small samples are flagged instead of guessed at."
    )

elif page == "Compare Players":
    st.subheader("Compare Two Players")

    seasons = metrics.get_recent_seasons()
    season = st.sidebar.selectbox("Season", seasons, index=0)

    player1_name = st.sidebar.selectbox(
        "Player 1", player_names, index=metrics.safe_selectbox_index(player_names, "LeBron James")
    )
    player2_name = st.sidebar.selectbox(
        "Player 2", player_names, index=metrics.safe_selectbox_index(player_names, "Kevin Durant")
    )

    player1_id = metrics.get_player_id(player_index, player1_name)
    player2_id = metrics.get_player_id(player_index, player2_name)

    try:
        df1 = data.fetch_player_game_log(player1_id, season)
        df2 = data.fetch_player_game_log(player2_id, season)
        advanced_df = data.fetch_advanced_stats(season)
    except data.PlayerStatsFetchError as exc:
        st.error(str(exc))
        st.stop()

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

    seasons = metrics.get_recent_seasons()
    season = st.sidebar.selectbox("Season", seasons, index=0, key="insights_season")
    player_name = st.sidebar.selectbox(
        "Player",
        player_names,
        index=metrics.safe_selectbox_index(player_names, "LeBron James"),
        key="insights_player",
    )
    player_id = metrics.get_player_id(player_index, player_name)

    try:
        game_log_df = data.fetch_player_game_log(player_id, season)
        all_teams = data.get_all_teams()
        team_advanced_df = data.fetch_team_advanced_stats(season)
        shot_df = data.fetch_shot_chart(player_id, season)
    except data.PlayerStatsFetchError as exc:
        st.error(str(exc))
        st.stop()

    if game_log_df.empty:
        st.error(f"No {season} game data for {player_name}.")
        st.stop()

    tab_absence, tab_defense, tab_hot_hand, tab_shot_chart = st.tabs(
        ["Absence & Rest Impact", "Performance vs. Defense", "Hot Hand Fallacy", "Shot Chart"]
    )

    with tab_absence:
        st.write(
            "No free live injury-designation feed exists anywhere, so this measures "
            "**games missed and return-game performance** -- never *why* a game was missed."
        )
        # PlayerGameLog has no TEAM_ID column -- derive it from the player's own
        # MATCHUP abbreviation instead (first token, before "vs."/"@").
        own_abbr = defense_tiers.extract_own_team_abbreviation(game_log_df.iloc[0]["MATCHUP"])
        team_id = metrics.build_abbreviation_to_team_id_index(all_teams).get(own_abbr)

        try:
            team_game_dates = data.fetch_team_game_dates(team_id, season) if team_id else []
        except data.PlayerStatsFetchError as exc:
            st.error(str(exc))
            team_game_dates = []

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
