import altair as alt
import numpy as np
import pandas as pd
import pytest

import charts


def _sample_game_log():
    return pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2023-10-24", "2024-01-15"]),
            "MATCHUP": ["LAL vs. HOU", "LAL @ DAL"],
            "PTS": [20, 25],
            "REB": [6, 7],
            "AST": [5, 6],
            "MIN": [34.0, 35.0],
        }
    )


def test_melt_for_overlay_row_count_and_player_column():
    df = _sample_game_log()
    melted = charts.melt_for_overlay(df, "LeBron James")
    # 2 games * 3 stat cols = 6 rows
    assert len(melted) == 6
    assert set(melted["Player"]) == {"LeBron James"}
    assert set(melted["Stat"]) == {"PTS", "REB", "AST"}
    assert "MATCHUP" not in melted.columns


def _two_players():
    return pd.concat(
        [
            charts.melt_for_overlay(_sample_game_log(), "LeBron James"),
            charts.melt_for_overlay(_sample_game_log(), "Kevin Durant"),
        ]
    )


def test_one_stat_game_log_colors_by_player_with_hoverable_dots():
    chart = charts.build_game_log_chart(_two_players(), ["LeBron James", "Kevin Durant"], ["PTS"])
    assert isinstance(chart, alt.LayerChart)
    lines, dots = (layer.to_dict() for layer in chart.layer)
    assert lines["mark"]["type"] == "line" and dots["mark"]["type"] == "circle"
    # Player 1 stays blue even though "Kevin Durant" sorts first alphabetically.
    assert lines["encoding"]["color"]["field"] == "Player"
    assert lines["encoding"]["color"]["scale"]["domain"] == ["LeBron James", "Kevin Durant"]
    assert lines["encoding"]["y"]["title"] == "Points"
    assert "tooltip" in dots["encoding"]


def test_all_three_stats_color_by_stat_never_by_player_color():
    one_player = charts.melt_for_overlay(_sample_game_log(), "LeBron James")
    chart = charts.build_game_log_chart(one_player, ["LeBron James", "Kevin Durant"], ["PTS", "REB", "AST"])
    lines = chart.layer[0].to_dict()
    assert lines["encoding"]["color"]["field"] == "Stat"
    assert lines["encoding"]["color"]["scale"]["range"] == list(charts.STAT_COLORS.values())
    assert not set(charts.STAT_COLORS.values()) & {charts.SERIES_BLUE, charts.SERIES_ORANGE}


def test_two_players_compare_one_stat_at_a_time():
    with pytest.raises(ValueError, match="one stat at a time"):
        charts.build_game_log_chart(_two_players(), ["LeBron James", "Kevin Durant"], ["PTS", "REB"])


def test_side_by_side_panel_shares_y_axis_and_has_no_player_legend():
    panel = charts.build_game_log_chart(
        charts.melt_for_overlay(_sample_game_log(), "Kevin Durant"),
        ["LeBron James", "Kevin Durant"],
        ["REB"],
        y_max=50,
        date_range=(pd.Timestamp("2023-10-01"), pd.Timestamp("2024-04-30")),
        player_legend=False,
    )
    lines = panel.layer[0].to_dict()
    assert lines["encoding"]["y"]["scale"]["domain"] == [0, 50]
    assert lines["encoding"]["x"]["scale"]["domain"] == [
        {"year": 2023, "month": 10, "date": 1},
        {"year": 2024, "month": 4, "date": 30},
    ]
    assert lines["encoding"]["color"]["legend"] is None


def test_build_null_distribution_chart_returns_layered_chart():
    null_dist = np.array([-0.02, 0.01, 0.03, -0.01, 0.0])
    chart = charts.build_null_distribution_chart(null_dist, observed_diff=0.05)
    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 2
    # smoke test: full spec resolves without raising (data + encodings are consistent)
    assert "layer" in chart.to_dict()


def test_build_delta_bar_chart_encodes_stat_and_delta():
    stats = {
        "PTS": {"return_avg": 20.5, "baseline_avg": 21.15, "delta": -0.65},
        "AST": {"return_avg": 7.75, "baseline_avg": 6.92, "delta": 0.83},
    }
    chart = charts.build_delta_bar_chart(stats)
    assert isinstance(chart, alt.Chart)
    assert chart.encoding.x.shorthand == "Delta:Q"
    assert chart.encoding.y.shorthand == "Stat:N"
    assert set(chart.data["Stat"]) == {"PTS", "AST"}
    # smoke test: full spec resolves without raising
    assert "encoding" in chart.to_dict()


def test_build_backtest_chart_layers_and_series_colors():
    dates = pd.to_datetime(["2025-11-01", "2025-11-03", "2025-11-05"])
    long_df = pd.concat(
        [
            pd.DataFrame({"GAME_DATE": dates, "Series": label, "Value": values})
            for label, values in [
                ("Actual", [20, 31, 18]),
                ("Model", [22.0, 22.5, 24.1]),
                ("Baseline (10-game avg)", [21.0, 21.4, 22.9]),
            ]
        ]
    )
    order = ["Actual", "Model", "Baseline (10-game avg)"]
    chart = charts.build_backtest_chart(long_df, order, y_title="PTS")
    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 3  # actual dots, prediction lines, hover crosshair
    spec = chart.to_dict()
    color_scale = spec["layer"][0]["encoding"]["color"]["scale"]
    assert color_scale["domain"] == order
    assert color_scale["range"] == [charts.SERIES_NEUTRAL, charts.SERIES_BLUE, charts.SERIES_ORANGE]


def test_build_contract_value_chart_highlights_both_players_in_fixed_slots():
    values = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "PLAYER_NAME": ["A", "B", "C", "D"],
            "SALARY": [50e6, 2e6, 20e6, 8e6],
            "IMPLIED_SALARY": [30e6, 6e6, 21e6, 7e6],
            "LABEL": ["Paid above production", "Paid below production", "Fairly paid", "Fairly paid"],
        }
    )
    chart = charts.build_contract_value_chart(values, {3: "C", 1: "A"}, typical_ratio=1.5)
    assert isinstance(chart, alt.LayerChart)
    assert len(chart.layer) == 6  # fair band, diagonal, league dots, highlighted players, two labels
    color = chart.to_dict()["layer"][3]["encoding"]["color"]["scale"]
    assert color == {"domain": ["C", "A"], "range": [charts.SERIES_BLUE, charts.SERIES_ORANGE]}

    # Player 2 keeps orange even when Player 1 isn't on the chart.
    unplotted = charts.build_contract_value_chart(values, {99: "Not Plotted", 1: "A"}, typical_ratio=1.5)
    assert unplotted.to_dict()["layer"][3]["encoding"]["color"]["scale"]["domain"] == ["Not Plotted", "A"]


def test_build_defense_tier_chart_encodes_tier_and_value():
    tier_result = pd.DataFrame(
        {"PTS": [19.3, 22.1], "REB": [6.0, 6.1], "AST": [6.3, 7.8], "Games": [25, 35]},
        index=pd.Index(["Elite Defense", "Weak Defense"], name="Tier"),
    )
    chart = charts.build_defense_tier_chart(tier_result)
    assert isinstance(chart, alt.Chart)
    assert chart.encoding.x.shorthand == "Stat:N"
    assert chart.encoding.xOffset.shorthand == "Tier:N"
    assert chart.encoding.y.shorthand == "Value:Q"
    assert chart.encoding.color.shorthand == "Tier:N"
    assert set(chart.data["Stat"]) == {"PTS", "REB", "AST"}
    # smoke test: full spec resolves without raising
    assert "encoding" in chart.to_dict()
