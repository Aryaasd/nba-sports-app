"""Chart-data prep and Altair chart builders.

Colors come from the validated dark-mode reference palette (dataviz skill,
references/palette.md): fixed categorical hue slots for series identity, a
blue/red diverging pair for polarity (deltas), reserved apart from both. A
matching Altair theme is registered so every chart's chrome (surface, grid,
text) sits on the same dark surface as the rest of the app instead of
rendering as a stray white rectangle against it.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

import metrics

# Categorical slots 1 & 2 (dark mode) -- player identity, fixed order, never cycled.
SERIES_BLUE = "#3987e5"
SERIES_ORANGE = "#d95926"

# Diverging pair (dark mode) -- polarity around a baseline (e.g. return-game deltas).
DIVERGING_NEGATIVE = "#3987e5"  # below baseline
DIVERGING_POSITIVE = "#e66767"  # above baseline
DIVERGING_NEUTRAL = "#383835"

_DARK_SURFACE = "#1a1a19"
_DARK_TEXT_PRIMARY = "#ffffff"
_DARK_TEXT_SECONDARY = "#c3c2b7"
_DARK_GRIDLINE = "#2c2c2a"
_DARK_AXIS = "#383835"


@alt.theme.register("nba_dark", enable=True)
def _nba_dark_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": _DARK_SURFACE,
                "view": {"stroke": "transparent"},
                "title": {"color": _DARK_TEXT_PRIMARY},
                "axis": {
                    "labelColor": _DARK_TEXT_SECONDARY,
                    "titleColor": _DARK_TEXT_PRIMARY,
                    "gridColor": _DARK_GRIDLINE,
                    "domainColor": _DARK_AXIS,
                    "tickColor": _DARK_AXIS,
                },
                "legend": {"labelColor": _DARK_TEXT_SECONDARY, "titleColor": _DARK_TEXT_PRIMARY},
                "header": {"labelColor": _DARK_TEXT_SECONDARY, "titleColor": _DARK_TEXT_PRIMARY},
            }
        }
    )


def prepare_line_chart_data(df: pd.DataFrame, stat_cols: list[str] | None = None) -> pd.DataFrame:
    """GAME_DATE-indexed df of just the stat columns, ready for st.line_chart."""
    stat_cols = stat_cols or metrics.DEFAULT_STAT_COLS
    return df.set_index("GAME_DATE")[list(stat_cols)]


def melt_for_overlay(df: pd.DataFrame, player_name: str, stat_cols: list[str] | None = None) -> pd.DataFrame:
    """Long-format df (GAME_DATE, Stat, Value, Player) for the Altair overlay chart."""
    stat_cols = stat_cols or metrics.DEFAULT_STAT_COLS
    melted = df[["GAME_DATE", *stat_cols]].melt("GAME_DATE", var_name="Stat", value_name="Value")
    melted["Player"] = player_name
    return melted


def build_overlay_chart(combined_df: pd.DataFrame) -> alt.Chart:
    """Two-player overlay: color by player (fixed categorical slots 1 & 2),
    line style by stat, returned unrendered."""
    return (
        alt.Chart(combined_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x="GAME_DATE:T",
            y="Value:Q",
            color=alt.Color("Player:N", scale=alt.Scale(range=[SERIES_BLUE, SERIES_ORANGE])),
            strokeDash="Stat:N",  # different line style for PTS/REB/AST
            tooltip=["GAME_DATE:T", "Player", "Stat", "Value"],
        )
        .properties(width=900, height=450)
    )


def build_null_distribution_chart(null_distribution: np.ndarray, observed_diff: float) -> alt.LayerChart:
    """Histogram of the hot-hand permutation test's null distribution (blue --
    what chance alone produces) with a rule marking where the observed
    statistic actually fell (red -- the diverging pair, reused here for "the
    single real observation standing apart from the simulated population")."""
    hist = (
        alt.Chart(pd.DataFrame({"diff": null_distribution}))
        .mark_bar(opacity=0.8, color=SERIES_BLUE)
        .encode(
            x=alt.X("diff:Q", bin=alt.Bin(maxbins=40), title="Shuffled make-rate difference"),
            y=alt.Y("count()", title="Permutations"),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"observed": [observed_diff]}))
        .mark_rule(color=DIVERGING_POSITIVE, strokeWidth=2)
        .encode(x="observed:Q")
    )
    # No fixed width: rendered with use_container_width=True. A quantitative
    # x-axis just gains resolution when stretched, nothing distorts.
    return (hist + rule).properties(height=320)


def build_delta_bar_chart(stats: dict[str, dict[str, float]]) -> alt.Chart:
    """Diverging bar chart for return-game deltas: one bar per stat, colored by
    polarity (below/above the player's own season baseline) around a zero
    midpoint -- the canonical diverging use case, not decoration."""
    df = pd.DataFrame(
        [{"Stat": stat, "Delta": values["delta"]} for stat, values in stats.items()]
    )
    return (
        alt.Chart(df)
        .mark_bar(size=28, cornerRadiusEnd=4)
        .encode(
            x=alt.X("Delta:Q", title="Return-game avg minus season baseline"),
            y=alt.Y("Stat:N", sort=None, title=None),
            color=alt.condition(
                "datum.Delta >= 0",
                alt.value(DIVERGING_POSITIVE),
                alt.value(DIVERGING_NEGATIVE),
            ),
            tooltip=["Stat", alt.Tooltip("Delta:Q", format="+.2f")],
        )
        .properties(height=200)
    )


def build_defense_tier_chart(tier_result: pd.DataFrame) -> alt.Chart:
    """Grouped bars: PTS/REB/AST per opponent-defense tier, colored by tier.

    Uses xOffset for grouping (one chart, clustered bars) rather than Altair's
    `column` facet -- a faceted chart is a compound view with its own per-facet
    width and ignores use_container_width, so it overflowed its column instead
    of scaling to fit.
    """
    df = tier_result.reset_index()
    melted = df.melt(id_vars=["Tier"], value_vars=["PTS", "REB", "AST"], var_name="Stat", value_name="Value")
    return (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Stat:N", title=None),
            xOffset=alt.XOffset("Tier:N"),
            y=alt.Y("Value:Q", title="Per-game average"),
            color=alt.Color("Tier:N", scale=alt.Scale(range=[SERIES_BLUE, SERIES_ORANGE]), title="Opponent"),
            tooltip=["Tier", "Stat", alt.Tooltip("Value:Q", format=".1f")],
        )
        .properties(height=280)
    )
