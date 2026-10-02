import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import about_page
import charts
import data
import metrics
import modeling
import sample_data_loader
import ui
from insights import absence, contract_value, defense_tiers, forecast, hot_hand, shot_chart

st.set_page_config(page_title="NBA Sports App", layout="wide")
st.html(ui.global_css())

PAGES = ["Home", "Compare Players", "Insights", "Predictions", "About"]
st.sidebar.html(ui.brand())
page = st.sidebar.radio("Navigate", PAGES, key="nav", label_visibility="collapsed", width="stretch")
# Each page's own selectors go here, so the credits below stay at the bottom of the sidebar.
sidebar_controls = st.sidebar.container()
st.sidebar.divider()
st.sidebar.html(
    ui.credits(["Stats: nba_api (NBA.com)", "Salaries: BALLDONTLIE, 2025-26", "Headshots: NBA.com"])
)

# Compared players are drawn in their team colors (ui.player_line_colors); these slot colors
# stand in when a team is unknown or two teams' colors are too close to tell apart.
DEFAULT_PLAYER_COLORS = [charts.SERIES_BLUE, charts.SERIES_ORANGE]
SAMPLE_LABEL = (
    f"{' / '.join(sample_data_loader.SAMPLE_PLAYERS.values())}, {sample_data_loader.SAMPLE_SEASON}"
)
# 10 games of history before the first prediction, plus enough predictions to judge.
MIN_PREDICTION_GAMES = 15
# The Home page's hot-hand finding uses this bundled player-season, so it never needs the API.
FEATURED_PLAYER_ID = 2544
ALL_STATS = "All three"
STAT_CHOICE_KEYS = {"Side by side": "compare_stat_side", "Overlay": "compare_stat_overlay"}
GO_TO_LABELS = {
    "Compare Players": "Compare players",
    "Insights": "Open insights",
    "Predictions": "See predictions",
    "About": "How it all works",
}


@st.cache_data(ttl=data.CACHE_TTL_SECONDS, show_spinner="Running the permutation test...")
def _cached_hot_hand_summary(shot_df: pd.DataFrame, n_permutations: int = 2000, seed: int = 42) -> dict:
    # The permutation test itself is pure computation (no nba_api call), but at
    # ~1s for 2000 shuffles it's worth caching same as a network fetch -- Streamlit
    # reruns this whole script on every widget interaction on the Insights page.
    return hot_hand.summarize_hot_hand(shot_df, n_permutations=n_permutations, seed=seed)


@st.cache_data(ttl=data.CACHE_TTL_SECONDS, show_spinner="Fitting the forecast...")
def _cached_forecast_summary(game_log_df: pd.DataFrame, stat_col: str) -> dict:
    return forecast.summarize_forecast(game_log_df, stat_col)


def _sidebar_select(label: str, options: list[str], key: str, default: str) -> str:
    # Default via session_state, not `index=`: Streamlit warns when a widget has a
    # non-default index *and* a callback (the sample-data button) writes its key.
    if key not in st.session_state:
        st.session_state[key] = options[metrics.safe_selectbox_index(options, default)]
    return sidebar_controls.selectbox(label, options, key=key)


def _sync_stat_choice() -> None:
    """Carry the Compare stat across a view switch. Overlay has no "All three", so switching
    to it from there starts Overlay on Points."""
    side, overlay = (st.session_state.get(key) for key in STAT_CHOICE_KEYS.values())
    if st.session_state["compare_view"] == "Overlay":
        if side and side != ALL_STATS:
            st.session_state[STAT_CHOICE_KEYS["Overlay"]] = side
    elif overlay:
        st.session_state[STAT_CHOICE_KEYS["Side by side"]] = overlay


def _go_to(page_name: str) -> None:
    # A callback, like _select_sample_data: the nav radio has already rendered this run.
    st.session_state["nav"] = page_name


def _select_sample_data(season_key: str, player_keys: list[str]) -> None:
    # Must run as an on_click callback: writing a widget's session_state after
    # the widget has rendered in the same run raises StreamlitAPIException.
    st.session_state[season_key] = sample_data_loader.SAMPLE_SEASON
    for key, name in zip(player_keys, sample_data_loader.SAMPLE_PLAYERS.values()):
        st.session_state[key] = name


def _render_contract_value(compared: dict[int, str], colors: list[str]) -> None:
    """Needs no live data (it reads a committed snapshot), so it renders even when the stats
    API is unreachable -- and for any selected season, labeled with the one it covers."""
    season = contract_value.CONTRACT_SEASON
    st.html(ui.section_header("Contract value", f"{season} salaries"))
    values = contract_value.load_contract_values()
    summary = contract_value.load_model_summary()
    if values is None or summary is None:
        st.info("Contract values haven't been built yet -- run `python -m scripts.train_contract_model`.")
        return
    shipped = summary["shipped_model"]
    typical_ratio = summary["cv_metrics"][shipped]["typical_ratio"]
    # A partial-season cap hit isn't a salary, so it doesn't belong on a salary axis.
    plotted = values[values["QUALIFIED"] & ~values["PARTIAL_DEAL"]]

    verdict_col, chart_col = st.columns([2, 3])
    with verdict_col:
        for (player_id, name), color in zip(compared.items(), colors):
            st.html(ui.player_label(name, color))
            match = values[values["PLAYER_ID"] == player_id]
            if match.empty:
                st.caption(
                    f"No {season} contract on file: he didn't play that season, or was on a two-way "
                    "deal, which the data source doesn't include."
                )
                continue
            row = match.iloc[0]
            figures = [
                (
                    "Cap hit (partial season)" if row.PARTIAL_DEAL else "Salary",
                    contract_value.format_millions(row.SALARY),
                )
            ]
            # Only figures the model stands behind: not for players it declines to value.
            if row.LABEL in contract_value.VALUED_LABELS:
                figures.append(("Production implies", contract_value.format_millions(row.IMPLIED_SALARY)))
            st.html(ui.figures(figures))
            result = {"label": row.LABEL, "ratio": row.RATIO}
            sentence = contract_value.describe_verdict(result, row.IMPLIED_SALARY)
            # Escaped: Streamlit markdown reads text between two "$" signs as LaTeX math.
            escaped = sentence.replace("$", r"\$")
            st.write(f"**{row.LABEL}.** {escaped}")
    with chart_col:
        st.altair_chart(
            charts.build_contract_value_chart(plotted, compared, typical_ratio, colors),
            use_container_width=True,
            theme=None,
        )
        missing = [name for player_id, name in compared.items() if player_id not in set(plotted["PLAYER_ID"])]
        if missing:
            st.caption(f"Not plotted: {' or '.join(missing)} (no full-season contract with enough minutes).")
    st.caption(
        f"Production-implied salary: a {contract_value.MODEL_LABELS[shipped]} of log salary on minutes, "
        "scoring, rebounding, assists, usage, true shooting, PIE, and age, learned from "
        f"{summary['n_market_priced']} market-priced contracts plus {summary['n_max_deals']} max deals. "
        "Rookie-scale, minimum-level, and partial-season deals don't follow market prices, so they're "
        "valued but never judged. Each player is valued by a model that never saw his own salary, and a "
        f"gap within its typical miss ({typical_ratio:.2f}x) counts as fairly paid. Box scores capture "
        "defense poorly, so defensive specialists can look overpaid, and contracts price past and "
        f"expected seasons, not just this one. Salaries: {summary['source']}."
    )


def _contract_pill(player_id: int) -> str | None:
    values = contract_value.load_contract_values()
    match = values[values["PLAYER_ID"] == player_id] if values is not None else []
    if len(match) == 0:
        return None
    row = match.iloc[0]
    return f"{row.LABEL} · {contract_value.format_millions(row.SALARY)}"


def _team_of(game_log: pd.DataFrame | None) -> str | None:
    """The team from a player's latest game (in case he was traded mid-season), if any."""
    if game_log is None or game_log.empty:
        return None
    return defense_tiers.extract_own_team_abbreviation(game_log.iloc[-1]["MATCHUP"])


def _player_colors(game_logs: list[pd.DataFrame | None]) -> list[str]:
    """Each compared player's color on the cards and in every chart: his team's."""
    return ui.player_line_colors([_team_of(log) for log in game_logs], DEFAULT_PLAYER_COLORS)


def _player_card(
    player_id: int, name: str, season: str, game_log: pd.DataFrame | None, **card_args
) -> str:
    """A player card with his season line, or a note when there's no game log to summarize
    (`None` means the fetch failed; an empty log means he didn't play)."""
    if game_log is None:
        return ui.player_card(player_id, name, note="Live stats unavailable right now.", **card_args)
    if game_log.empty:
        return ui.player_card(player_id, name, note=f"No {season} games logged.", **card_args)
    return ui.player_card(
        player_id,
        name,
        team=_team_of(game_log),
        games=len(game_log),
        averages=metrics.season_averages(game_log),
        **card_args,
    )


def _render_faceoff(players: list[tuple[int, str]], season: str, game_logs: list, colors: list[str]) -> None:
    # Contract verdicts describe the 2025-26 snapshot; on any other season they'd mislead.
    show_pill = season == contract_value.CONTRACT_SEASON
    left, middle, right = st.columns([1, 0.1, 1], vertical_alignment="center")
    for col, (player_id, name), log, color in zip([left, right], players, game_logs, colors):
        pill = _contract_pill(player_id) if show_pill else None
        col.html(_player_card(player_id, name, season, log, series_color=color, pill=pill))
    middle.html(ui.versus())


def _render_page_header(eyebrow: str, title: str, blurb: str) -> DeltaGenerator:
    """Title block on the left; returns the right-hand column for the page's player card."""
    head_col, card_col = st.columns([1.1, 1], vertical_alignment="center", gap="large")
    head_col.html(ui.page_header(eyebrow, title, blurb))
    return card_col


def _render_findings() -> None:
    """Three headline results from the app's own analyses: two computed from committed data,
    one quoted from scripts/evaluate_forecast.py's league-wide run."""
    cards = []

    values = contract_value.load_contract_values()
    bargain = contract_value.biggest_bargain(values) if values is not None else None
    if bargain is not None:
        salary = contract_value.format_millions(bargain.SALARY)
        implied = contract_value.format_millions(bargain.IMPLIED_SALARY)
        body = (
            f"{bargain.PLAYER_NAME} scored {bargain.PTS:.1f} a night on {salary}; his production "
            f"prices at {implied}. The widest gap among the contracts the model judges."
        )
        cards.append((ui.finding_card("Contract value", salary, body, versus=implied), "Compare Players"))

    name = sample_data_loader.SAMPLE_PLAYERS[FEATURED_PLAYER_ID]
    shots = sample_data_loader.load_shot_chart(FEATURED_PLAYER_ID, sample_data_loader.SAMPLE_SEASON)
    result = _cached_hot_hand_summary(shots) if shots is not None else {"insufficient_data": True}
    if not result["insufficient_data"]:
        body = (
            f"{name} made {result['p_make_after_make']:.1%} of shots after a make and "
            f"{result['p_make_after_miss']:.1%} after a miss. In {result['n_permutations']:,} "
            f"shuffles of his own shot order, {result['p_value']:.0%} showed a gap at least that large."
        )
        cards.append((ui.finding_card("Hot hand", f"p = {result['p_value']:.2f}", body), "Insights"))

    league = forecast.LEAGUE_RESULT
    body = (
        "How often exponential smoothing out-forecast a plain season average, across "
        f"{league['n_players']} players' {league['season']} scoring. Its median weight on recent "
        f"games is {league['median_alpha_pts']:.0f}: recent form adds little."
    )
    beats = f"{league['ses_beats_season_mean_pts']:.0%}"
    cards.append((ui.finding_card("Season forecast", beats, body), "Predictions"))

    for col, (card, target) in zip(st.columns(len(cards)), cards):
        col.html(card)
        col.button(
            f"{GO_TO_LABELS[target]} →",
            key=f"goto_{target.lower().replace(' ', '_')}",
            on_click=_go_to,
            args=(target,),
        )


def _show_fetch_error(exc: Exception, season_key: str, player_keys: list[str]) -> None:
    if data.LIVE_API_ENABLED:
        st.error(
            "Couldn't load live NBA data for this selection -- the NBA stats API didn't respond. "
            "It blocks many cloud servers and briefly throttles bursts of requests, so a local "
            "run may just need a few minutes."
        )
    else:
        st.error(
            "This selection isn't in the cached snapshot. The hosted demo can't use live NBA data: "
            "the NBA stats API blocks its servers. Run the app locally for every player and season."
        )
    st.button(
        f"Explore with cached sample data ({SAMPLE_LABEL})",
        on_click=_select_sample_data,
        args=(season_key, player_keys),
        type="primary",
    )
    with st.expander("Error details"):
        st.code(str(exc), language=None)


def _stop_with_fetch_error(exc: Exception, season_key: str, player_keys: list[str]) -> None:
    _show_fetch_error(exc, season_key, player_keys)
    st.stop()


def _stop_if_no_games(game_logs: dict[str, pd.DataFrame], season: str) -> None:
    # Not an error: the fetch worked, there are just no games -- e.g. a newly listed
    # season that hasn't tipped off, or a player who didn't play that year.
    missing = [name for name, log in game_logs.items() if log.empty]
    if missing:
        st.info(f"No {season} games logged for {' or '.join(missing)} yet. Pick another season or player.")
        st.stop()


def _show_sample_banner(*used_sample_flags: bool) -> None:
    # Worded to stay true when only a league-wide table fell back for a player whose own
    # game log loaded live -- the snapshot's league tables cover every player.
    if any(used_sample_flags):
        players = " and ".join(sample_data_loader.SAMPLE_PLAYERS.values())
        why = (
            "the live NBA stats API is unreachable from this host right now"
            if data.LIVE_API_ENABLED
            else "the NBA stats API blocks this hosted demo's servers"
        )
        st.warning(
            f"Some of this page comes from a cached {sample_data_loader.SAMPLE_SEASON} snapshot of "
            f"real data -- {why}. The snapshot covers {players} plus league-wide tables; any other "
            "player shows an error rather than made-up data."
        )


try:
    all_players = data.get_all_players()
except data.PlayerStatsFetchError as exc:
    st.error(str(exc))
    st.stop()

player_index = metrics.build_player_index(all_players)
player_names = metrics.get_player_names(all_players)
seasons = metrics.get_recent_seasons()
# Without live data (the hosted demo), only the snapshot's season has anything to show.
default_season = metrics.get_default_season() if data.LIVE_API_ENABLED else sample_data_loader.SAMPLE_SEASON

if page == "Home":
    st.html(
        ui.hero(
            f"NBA analytics, {sample_data_loader.SAMPLE_SEASON}",
            ["The box score,", "cross-examined"],
            "Compare players, test the hot hand, and price a contract. Every model here is scored "
            "against the simple baseline it has to beat.",
        )
    )

    st.html(ui.section_header("From the data", f"{sample_data_loader.SAMPLE_SEASON} regular season"))
    _render_findings()

    st.html(ui.section_header("What's inside"))
    tiles = [
        (
            "Compare players",
            "Two players' game logs side by side or overlaid, advanced rates (TS%, USG%, PACE, PIE) "
            "with per-36 production, and whether each is paid more or less than his production implies.",
        ),
        (
            "Insights",
            "Four tests for one player-season: return from absence, elite vs. weak defenses, a "
            "permutation test of the hot hand, and a shot-chart efficiency heatmap.",
        ),
        (
            "Predictions",
            "A next-game points model trained on 20,000+ player-games and a season forecast by "
            "exponential smoothing, both scored only on games they hadn't seen.",
        ),
        (
            "Ground rules",
            "Each analysis says what it can't show: no free injury feed exists, defensive-scheme "
            "data is proprietary, small samples are flagged, and every model sits beside its baseline.",
        ),
    ]
    for col, (title, body) in zip(st.columns(len(tiles), gap="medium"), tiles):
        col.html(ui.tile(title, body))
    st.button(f"{GO_TO_LABELS['About']} →", key="goto_about", on_click=_go_to, args=("About",))

elif page == "Compare Players":
    season = _sidebar_select("Season", seasons, "compare_season", default_season)

    # Curry, not a third star, by default: with LeBron he's in the bundled sample, so the
    # hosted demo opens on a working comparison even when the live API is blocked.
    player1_name = _sidebar_select("Player 1", player_names, "compare_player1", "LeBron James")
    player2_name = _sidebar_select("Player 2", player_names, "compare_player2", "Stephen Curry")

    player1_id = metrics.get_player_id(player_index, player1_name)
    player2_id = metrics.get_player_id(player_index, player2_name)
    players = [(player1_id, player1_name), (player2_id, player2_name)]
    compared = dict(players)

    st.html(
        ui.page_header(
            f"{season} regular season",
            "Compare players",
            "Two seasons, game by game, then the rates behind the box score and what each contract buys.",
        )
    )

    df1 = df2 = None  # stays None for a player whose game log didn't load
    try:
        df1, sample1 = data.fetch_player_game_log(player1_id, season)
        df2, sample2 = data.fetch_player_game_log(player2_id, season)
        advanced_df, sample_advanced = data.fetch_advanced_stats(season)
    except data.PlayerStatsFetchError as exc:
        colors = _player_colors([df1, df2])
        _render_faceoff(players, season, [df1, df2], colors)
        _show_fetch_error(exc, "compare_season", ["compare_player1", "compare_player2"])
        _render_contract_value(compared, colors)
        st.stop()

    _show_sample_banner(sample1, sample2, sample_advanced)
    colors = _player_colors([df1, df2])
    _render_faceoff(players, season, [df1, df2], colors)

    _stop_if_no_games({player1_name: df1, player2_name: df2}, season)

    st.html(ui.section_header("Game by game", "Every game, in order"))
    view_col, stat_col = st.columns([2, 3])
    view = view_col.radio(
        "Chart view",
        ["Side by side", "Overlay"],
        horizontal=True,
        key="compare_view",
        on_change=_sync_stat_choice,
    )
    # Overlay puts both players on one chart, so it compares one stat at a time; each view keeps
    # its own stat choice (synced on switching) since only side by side offers all three.
    choices = list(charts.STAT_NAMES.values()) + ([ALL_STATS] if view == "Side by side" else [])
    stat_choice = stat_col.radio("Stat", choices, horizontal=True, key=STAT_CHOICE_KEYS[view])
    stats = [code for code, name in charts.STAT_NAMES.items() if stat_choice in (name, ALL_STATS)]
    player_order = [player1_name, player2_name]

    if view == "Side by side":
        # Shared axes, so the same height means the same number and the same spot the same date.
        y_max = max(df[stats].to_numpy().max() for df in [df1, df2]) * 1.05
        dates = pd.concat([df1["GAME_DATE"], df2["GAME_DATE"]])
        for col, df, name in zip(st.columns(2), [df1, df2], player_order):
            col.altair_chart(
                charts.build_game_log_chart(
                    charts.melt_for_overlay(df, name),
                    player_order,
                    stats,
                    colors=colors,
                    y_max=y_max,
                    date_range=(dates.min(), dates.max()),
                    player_legend=False,
                ),
                use_container_width=True,
                theme=None,
            )
    else:
        combined_df = pd.concat(
            [
                charts.melt_for_overlay(df1, player1_name),
                charts.melt_for_overlay(df2, player2_name),
            ]
        )
        st.altair_chart(
            charts.build_game_log_chart(combined_df, player_order, stats, colors=colors, height=420),
            use_container_width=True,
            theme=None,
        )
    if len(stats) > 1:
        st.caption("Click a stat in the legend to show it alone; click empty space in the chart to reset.")

    st.html(ui.section_header("Advanced metrics", "Season rates"))
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
        table = metrics.build_comparison_table(player_stats)
        rates = pd.IndexSlice[["TS%", "USG%", "PIE"], :]  # shares, read to three places; the rest to one
        st.dataframe(
            table.style.format(precision=1, na_rep="N/A").format("{:.3f}", subset=rates, na_rep="N/A"),
            use_container_width=True,
        )
    with note_col:
        st.html(ui.subhead("Reading this table"))
        st.caption(
            "TS% and USG% come straight from `nba_api`'s Advanced measure type. "
            "PACE is the team's estimated possessions per 48 minutes while the "
            "player is on the floor. PIE is the league's own share-of-game-events "
            "metric. Per-36 rates are computed from this season's game log, not "
            "a separate API call."
        )

    _render_contract_value(compared, colors)

elif page == "Insights":
    season = _sidebar_select("Season", seasons, "insights_season", default_season)
    player_name = _sidebar_select("Player", player_names, "insights_player", "LeBron James")
    player_id = metrics.get_player_id(player_index, player_name)

    card_col = _render_page_header(
        f"{season} regular season",
        "Insights",
        "Four small, hypothesis-driven analyses for one player-season, honest about what the "
        "data can and can't show. None of them is a prediction.",
    )

    game_log_df = None
    try:
        game_log_df, sample_log = data.fetch_player_game_log(player_id, season)
        all_teams = data.get_all_teams()
        team_advanced_df, sample_team_stats = data.fetch_team_advanced_stats(season)
        shot_df, sample_shots = data.fetch_shot_chart(player_id, season)
    except data.PlayerStatsFetchError as exc:
        card_col.html(_player_card(player_id, player_name, season, game_log_df))
        _stop_with_fetch_error(exc, "insights_season", ["insights_player"])

    card_col.html(_player_card(player_id, player_name, season, game_log_df))
    _stop_if_no_games({player_name: game_log_df}, season)

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
                st.html(ui.subhead("Return-game vs. season baseline"))
                labels = {
                    "return_avg": "Return games",
                    "baseline_avg": "Season average",
                    "delta": "Difference",
                }
                deltas_df = pd.DataFrame(result["stats"]).T.rename(columns=labels)
                st.dataframe(
                    deltas_df.style.format("{:.2f}").format("{:+.2f}", subset=["Difference"]),
                    use_container_width=True,
                )
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
                st.html(ui.subhead("Per-game averages by opponent tier"))
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
                st.html(ui.subhead("Reading this test"))
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
            "render muted instead of a misleading 100%/0% from a single shot."
        )
        if shot_df.empty:
            st.info(f"No shot data for {player_name} this season.")
        else:
            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.altair_chart(shot_chart.build_shot_chart(shot_df), use_container_width=False, theme=None)
            with table_col:
                st.html(ui.subhead("Best zones this season"))
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
    season = _sidebar_select("Season", seasons, "predictions_season", default_season)
    player_name = _sidebar_select("Player", player_names, "predictions_player", "LeBron James")
    player_id = metrics.get_player_id(player_index, player_name)

    card_col = _render_page_header(
        f"{season} regular season",
        "Predictions",
        "Two forecasting techniques for one player-season. Each is scored only on games it "
        "hadn't seen yet, next to the simple averages it has to beat.",
    )

    try:
        game_log_df, sample_log = data.fetch_player_game_log(player_id, season)
    except data.PlayerStatsFetchError as exc:
        card_col.html(_player_card(player_id, player_name, season, None))
        _stop_with_fetch_error(exc, "predictions_season", ["predictions_player"])

    card_col.html(_player_card(player_id, player_name, season, game_log_df))
    _show_sample_banner(sample_log)

    _stop_if_no_games({player_name: game_log_df}, season)

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
                    st.html(ui.subhead(f"{player_name}, {season}: each prediction, made before the game"))
                    long_df = backtest.melt(
                        id_vars="GAME_DATE", value_vars=series, var_name="Series", value_name="Value"
                    )
                    st.altair_chart(
                        charts.build_backtest_chart(long_df, series, y_title="Points"),
                        use_container_width=True,
                        theme=None,
                    )
                with side_col:
                    st.html(ui.subhead(f"{player_name}'s typical miss this season"))
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
                    st.html(ui.subhead(f"Next game after {last_game:%b %d, %Y}"))
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
                st.html(ui.subhead("What the fitted weight says"))
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

elif page == "About":
    about_page.render()
