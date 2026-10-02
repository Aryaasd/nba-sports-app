"""Chart-data prep and Altair chart builders.

Data colors come from the validated dark-mode reference palette (dataviz skill,
references/palette.md): fixed categorical hue slots for series identity, a
blue/red diverging pair for polarity (deltas), reserved apart from both. A
matching Altair theme is registered so every chart's chrome (surface, grid,
text, fonts) uses the app's panel color and type (ui.css) instead of rendering
as a stray white rectangle against it.
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
DIVERGING_NEUTRAL = "#2F3E52"

# Game-log stats. Their colors are used only when all three share a chart, where color means
# stat rather than player, so none of them is the player blue or orange. They differ in
# lightness as well as hue (chalk, green, lilac), so they stay apart under color blindness.
STAT_NAMES = {"PTS": "Points", "REB": "Rebounds", "AST": "Assists"}
STAT_COLORS = {"PTS": "#F1ECE3", "REB": "#5CC98A", "AST": "#B48CF0"}

# Muted ink (dark mode) -- raw observations that should recede behind the predictions.
SERIES_NEUTRAL = "#8C96A3"

# Chart chrome, matching the app's panels and ink (ui.css: --seat, --chalk, --chalk-dim, --line).
_DARK_SURFACE = "#182230"
_DARK_TEXT_PRIMARY = "#F1ECE3"
_DARK_TEXT_SECONDARY = "#A7B0BC"
_DARK_GRIDLINE = "#223044"
_DARK_AXIS = "#2A3747"
_BODY_FONT = "Barlow, sans-serif"
_LABEL_FONT = "Barlow Condensed, Barlow, sans-serif"


@alt.theme.register("nba_dark", enable=True)
def _nba_dark_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": _DARK_SURFACE,
                "padding": 14,
                "font": _BODY_FONT,
                "view": {"stroke": "transparent"},
                "title": {"color": _DARK_TEXT_PRIMARY, "font": _LABEL_FONT},
                "axis": {
                    "labelColor": _DARK_TEXT_SECONDARY,
                    "labelFont": _BODY_FONT,
                    "labelFontSize": 12,
                    "titleColor": _DARK_TEXT_PRIMARY,
                    "titleFont": _LABEL_FONT,
                    "titleFontSize": 13,
                    "titleFontWeight": 600,
                    "gridColor": _DARK_GRIDLINE,
                    "domainColor": _DARK_AXIS,
                    "tickColor": _DARK_AXIS,
                },
                "legend": {
                    "labelColor": _DARK_TEXT_SECONDARY,
                    "labelFont": _BODY_FONT,
                    "labelFontSize": 13,
                    "titleColor": _DARK_TEXT_PRIMARY,
                    "titleFont": _LABEL_FONT,
                    "titleFontSize": 13,
                },
                "header": {"labelColor": _DARK_TEXT_SECONDARY, "titleColor": _DARK_TEXT_PRIMARY},
                "text": {"font": _BODY_FONT},
            }
        }
    )


def melt_for_overlay(df: pd.DataFrame, player_name: str, stat_cols: list[str] | None = None) -> pd.DataFrame:
    """Long-format df (GAME_DATE, Stat, Value, Player) for the Altair overlay chart."""
    stat_cols = stat_cols or metrics.DEFAULT_STAT_COLS
    melted = df[["GAME_DATE", *stat_cols]].melt("GAME_DATE", var_name="Stat", value_name="Value")
    melted["Player"] = player_name
    return melted


def build_game_log_chart(
    long_df: pd.DataFrame,
    player_order: list[str],
    stats: list[str],
    *,
    colors: list[str] | None = None,
    y_max: float | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    height: int = 360,
    player_legend: bool = True,
) -> alt.LayerChart:
    """Game-by-game values from melt_for_overlay rows, for one or both players, with a dot
    per game for hover details. Returned unrendered.

    - One stat: color means player: `colors` in `player_order` (the app passes team colors;
      the default is categorical slots 1 & 2). The explicit domain keeps each player's color
      even when names sort the other way.
    - Several stats, for one player only: color means stat (STAT_COLORS, none of them a
      player color). Clicking a stat in the legend isolates it. Two players are compared one
      stat at a time; with all three each, six lines would be unreadable.

    `y_max` and `date_range` pin the axes so side-by-side panels share both scales;
    `player_legend=False` for one-player panels, where the player cards above already carry
    the color key.
    """
    if len(stats) > 1 and long_df["Player"].nunique() > 1:
        raise ValueError("Compare two players one stat at a time.")
    df = long_df[long_df["Stat"].isin(stats)].assign(Stat=lambda d: d["Stat"].map(STAT_NAMES))
    names = [STAT_NAMES[stat] for stat in stats]
    x_scale = alt.Undefined
    if date_range:
        x_scale = alt.Scale(domain=[alt.DateTime(year=d.year, month=d.month, date=d.day) for d in date_range])
    x = alt.X("GAME_DATE:T", title=None, scale=x_scale)
    y_scale = alt.Scale(domain=[0, y_max], nice=False) if y_max else alt.Undefined
    y = alt.Y("Value:Q", title=names[0] if len(stats) == 1 else "Per game", scale=y_scale)
    tooltip = [
        alt.Tooltip("GAME_DATE:T", title="Game", format="%b %d, %Y"),
        alt.Tooltip("Player:N"),
        alt.Tooltip("Stat:N"),
        alt.Tooltip("Value:Q", format=".0f"),
    ]
    top_legend = alt.Legend(title=None, orient="top")

    if len(stats) == 1:
        base = alt.Chart(df).encode(
            x=x,
            y=y,
            color=alt.Color(
                "Player:N",
                scale=alt.Scale(domain=player_order, range=colors or [SERIES_BLUE, SERIES_ORANGE]),
                legend=top_legend if player_legend else None,
            ),
        )
        lines = base.mark_line(strokeWidth=2)
        dots = base.mark_circle(size=32, opacity=0.95).encode(tooltip=tooltip)
        return alt.layer(lines, dots).properties(width=900, height=height)

    picked = alt.selection_point(fields=["Stat"], bind="legend")
    base = alt.Chart(df).encode(
        x=x,
        y=y,
        color=alt.Color(
            "Stat:N", scale=alt.Scale(domain=names, range=[STAT_COLORS[s] for s in stats]), legend=top_legend
        ),
    )
    lines = base.mark_line(strokeWidth=1.75).encode(
        opacity=alt.condition(picked, alt.value(1), alt.value(0.1))
    ).add_params(picked)
    dots = base.mark_circle(size=22).encode(
        opacity=alt.condition(picked, alt.value(0.95), alt.value(0.06)), tooltip=tooltip
    )
    return alt.layer(lines, dots).properties(width=900, height=height)


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


def build_backtest_chart(long_df: pd.DataFrame, series_order: list[str], y_title: str) -> alt.LayerChart:
    """Actual per-game values vs. one-step-ahead predictions, from long (GAME_DATE, Series, Value) data.

    `series_order` is [actual, model, baseline]. Actuals are muted dots -- single games are
    noisy, and a zig-zag line would drown the two prediction lines the chart is about. The
    model is a solid line; the baseline is dashed, since it's a reference, not a result.
    """
    actual, model, baseline = series_order
    color = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=series_order, range=[SERIES_NEUTRAL, SERIES_BLUE, SERIES_ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X("GAME_DATE:T", title=None)
    base = alt.Chart(long_df).encode(x=x, y=alt.Y("Value:Q", title=y_title))

    dots = (
        base.transform_filter(alt.datum.Series == actual)
        .mark_circle(size=50, opacity=0.75)
        .encode(color=color)
    )
    lines = (
        base.transform_filter(alt.datum.Series != actual)
        .mark_line(strokeWidth=2)
        .encode(
            color=color,
            strokeDash=alt.StrokeDash(
                "Series:N", scale=alt.Scale(domain=[model, baseline], range=[[1, 0], [6, 4]]), legend=None
            ),
        )
    )

    # Full-height hover rule with every series' value for that game (Altair's multi-line tooltip pattern).
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["GAME_DATE"], empty=False)
    crosshair = (
        alt.Chart(long_df)
        .transform_pivot("Series", value="Value", groupby=["GAME_DATE"])
        .mark_rule(color=_DARK_TEXT_SECONDARY, strokeWidth=1)
        .encode(
            x=x,
            opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
            tooltip=[alt.Tooltip("GAME_DATE:T", title="Game")]
            + [alt.Tooltip(f"{name}:Q", format=".1f") for name in series_order],
        )
        .add_params(hover)
    )
    return (dots + lines + crosshair).properties(height=340)


def build_contract_value_chart(
    values_df: pd.DataFrame, highlight: dict[int, str], typical_ratio: float, colors: list[str] | None = None
) -> alt.LayerChart:
    """Actual salary vs. production-implied salary for every judged player, log-log.

    The shaded band is "fairly paid" (within the model's typical miss of the diagonal);
    dots above it are paid more than production implies, below it less. The two
    compared players keep the colors they have elsewhere on the page (`colors`, in
    `highlight` order; categorical slots 1 & 2 by default).
    """
    league = values_df.assign(Salary=values_df["SALARY"] / 1e6, Implied=values_df["IMPLIED_SALARY"] / 1e6)
    both = league[["Salary", "Implied"]]
    low, high = both.min().min() * 0.8, both.max().max() * 1.2
    # nice=False: Vega would otherwise round each log axis out to a power of ten on its own,
    # leaving the two axes on different ranges and the diagonal off-center.
    scale = alt.Scale(type="log", domain=[low, high], nice=False)
    x = alt.X("Implied:Q", scale=scale, title="Production implies ($M)")
    y = alt.Y("Salary:Q", scale=scale, title="Actual salary ($M)")

    implied = np.geomspace(low, high, 50)
    diagonal = pd.DataFrame({"Implied": implied, "Salary": implied})
    diagonal["Low"], diagonal["High"] = implied / typical_ratio, implied * typical_ratio
    # The band shares the explicit y scale and clips to it: its own low edge sits below the
    # data, and in a layered chart an unscaled layer would stretch everyone's y domain.
    band = alt.Chart(diagonal).mark_area(color=DIVERGING_NEUTRAL, opacity=0.5, clip=True)
    band = band.encode(x=x, y=alt.Y("Low:Q", scale=scale, title="Actual salary ($M)"), y2="High:Q")
    line = alt.Chart(diagonal).mark_line(color=_DARK_TEXT_SECONDARY, strokeDash=[6, 4], strokeWidth=1)
    line = line.encode(x=x, y=y)

    tooltip = [
        alt.Tooltip("PLAYER_NAME:N", title="Player"),
        alt.Tooltip("Salary:Q", format="$.1f", title="Salary ($M)"),
        alt.Tooltip("Implied:Q", format="$.1f", title="Production implies ($M)"),
        alt.Tooltip("LABEL:N", title="Verdict"),
    ]
    dots = alt.Chart(league).mark_circle(size=40, color=SERIES_NEUTRAL, opacity=0.55)
    dots = dots.encode(x=x, y=y, tooltip=tooltip)

    picked = league[league["PLAYER_ID"].isin(highlight)]
    picked = picked.assign(Player=picked["PLAYER_ID"].map(highlight))
    player_color = alt.Color(
        "Player:N",
        scale=alt.Scale(domain=list(highlight.values()), range=colors or [SERIES_BLUE, SERIES_ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    marked = alt.Chart(picked).mark_circle(size=160, opacity=1, stroke=_DARK_SURFACE, strokeWidth=2)
    marked = marked.encode(x=x, y=y, color=player_color, tooltip=tooltip)
    # Labels sit left of their dots (the best-paid players crowd the top-right edge, where a
    # right-hand label would clip), first player above and second below so close dots stay legible.
    label_layers = []
    for name, dy in zip(highlight.values(), [-12, 16]):
        text = alt.Chart(picked[picked["Player"] == name]).mark_text(
            align="right", dx=-12, dy=dy, fontSize=12, color=_DARK_TEXT_PRIMARY
        )
        label_layers.append(text.encode(x=x, y=y, text="Player:N"))
    return alt.layer(band, line, dots, marked, *label_layers).properties(height=380)


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
