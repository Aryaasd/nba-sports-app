"""Shot Chart Heatmap: FG% by spatial bin, layered over a hand-drawn court.

Shares the exact same `fetch_shot_chart` pull as the hot-hand test -- adding
this alongside it costs only rendering, not a new API call.

Deliberately stays on Altair (no new matplotlib dependency): the rest of the
app is Altair-only already, and Altair can draw a court fine via a small
constant reference table of line segments in the same LOC_X/LOC_Y coordinate
space (tenths of a foot, hoop at the origin) that nba_api's shot chart
endpoint uses -- these are the standard, widely-published NBA court
coordinates used throughout the basketball-analytics community.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

BIN_SIZE = 40  # ~4 feet per spatial bin
DEFAULT_MIN_ATTEMPTS = 3
# Dark, low-contrast so a "not enough data" bin visibly recedes against the
# app's dark chart surface instead of becoming the brightest thing on screen.
MUTED_COLOR = "#2F3E52"
COURT_LINE_COLOR = "#8C96A3"  # muted ink, matching charts.SERIES_NEUTRAL


def _arc_points(cx: float, cy: float, r: float, theta1_deg: float, theta2_deg: float, n: int = 50):
    thetas = np.radians(np.linspace(theta1_deg, theta2_deg, n))
    return cx + r * np.cos(thetas), cy + r * np.sin(thetas)


def _rect_points(x0: float, y0: float, width: float, height: float):
    return [x0, x0 + width, x0 + width, x0, x0], [y0, y0, y0 + height, y0 + height, y0]


def court_lines_df() -> pd.DataFrame:
    """NBA half-court boundary lines as (line_id, x, y) points, one row per
    vertex, grouped by `line_id` so each court element draws as its own
    connected line rather than one continuous scribble."""
    segments: list[dict] = []

    def add(name: str, xs, ys) -> None:
        # `seq` preserves each shape's path order (rectangle corners, arc sweep
        # direction) -- without it, mark_line's default point-connection order
        # would sort by x or y and scramble every non-monotonic shape.
        segments.extend(
            {"line_id": name, "seq": i, "x": x, "y": y} for i, (x, y) in enumerate(zip(xs, ys))
        )

    add("hoop", *_arc_points(0, 0, 7.5, 0, 360))
    add("backboard", [-30, 30], [-7.5, -7.5])
    add("outer_box", *_rect_points(-80, -47.5, 160, 190))
    add("paint", *_rect_points(-60, -47.5, 120, 190))
    add("free_throw_circle", *_arc_points(0, 142.5, 60, 0, 180))
    add("restricted_area", *_arc_points(0, 0, 40, 0, 180))
    add("corner_three_left", [-220, -220], [-47.5, 92.5])
    add("corner_three_right", [220, 220], [-47.5, 92.5])
    add("three_point_arc", *_arc_points(0, 0, 237.5, 22, 158))
    add("court_boundary", *_rect_points(-250, -47.5, 500, 470))

    return pd.DataFrame(segments)


def bin_shots(shot_df: pd.DataFrame, bin_size: int = BIN_SIZE) -> pd.DataFrame:
    """Aggregate shots into fixed-size spatial bins: attempts, makes, FG% per bin."""
    df = shot_df.copy()
    df["x_bin"] = (df["LOC_X"] // bin_size) * bin_size
    df["y_bin"] = (df["LOC_Y"] // bin_size) * bin_size
    grouped = (
        df.groupby(["x_bin", "y_bin"])
        .agg(attempts=("SHOT_MADE_FLAG", "size"), makes=("SHOT_MADE_FLAG", "sum"))
        .reset_index()
    )
    grouped["fg_pct"] = grouped["makes"] / grouped["attempts"]
    grouped["x0"] = grouped["x_bin"]
    grouped["x1"] = grouped["x_bin"] + bin_size
    grouped["y0"] = grouped["y_bin"]
    grouped["y1"] = grouped["y_bin"] + bin_size
    return grouped


def build_shot_chart(shot_df: pd.DataFrame, min_attempts: int = DEFAULT_MIN_ATTEMPTS) -> alt.LayerChart:
    """FG% per spatial bin, with bins under `min_attempts` muted grey instead of
    a misleading 100%/0% color from a single shot -- same "don't let small
    samples lie" discipline as the other three insights, applied to a visual."""
    binned = bin_shots(shot_df)

    court = (
        alt.Chart(court_lines_df())
        .mark_line(color=COURT_LINE_COLOR, strokeWidth=1.5)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), detail="line_id:N", order="seq:Q")
    )

    heat = (
        alt.Chart(binned)
        .mark_rect(opacity=0.85)
        .encode(
            x=alt.X("x0:Q", axis=None),
            x2="x1:Q",
            y=alt.Y("y0:Q", axis=None),
            y2="y1:Q",
            color=alt.condition(
                f"datum.attempts >= {min_attempts}",
                # Deliberate deviation from the palette's default sequential
                # blue: every real NBA shot chart uses hot=red/orange for FG%,
                # and inverting that near-universal convention would read as
                # backwards to anyone who's seen one before.
                alt.Color("fg_pct:Q", scale=alt.Scale(scheme="orangered", domain=[0, 1]), title="FG%"),
                alt.value(MUTED_COLOR),
            ),
            tooltip=[
                alt.Tooltip("attempts:Q", title="Attempts"),
                alt.Tooltip("makes:Q", title="Makes"),
                alt.Tooltip("fg_pct:Q", title="FG%", format=".1%"),
            ],
        )
    )

    return (heat + court).properties(width=500, height=470)


def label_zone(x0: float, y0: float, bin_size: int = BIN_SIZE) -> str:
    """Rough shot-zone label for a bin's center, by distance from the hoop.

    Not an official zone classification (real zone boundaries follow the
    court's straight lines, not a pure radius) -- a lightweight approximation
    for a supplementary "hot zones" summary, using the same radii already
    drawn for the restricted area (40) and three-point arc (237.5).
    """
    cx, cy = x0 + bin_size / 2, y0 + bin_size / 2
    distance = (cx**2 + cy**2) ** 0.5
    if distance < 40:
        return "Restricted Area"
    if distance < 237.5:
        return "Mid-Range"
    return "Three-Point"
