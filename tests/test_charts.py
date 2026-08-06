import altair as alt
import numpy as np
import pandas as pd

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


def test_prepare_line_chart_data_index_and_columns():
    result = charts.prepare_line_chart_data(_sample_game_log())
    assert result.index.name == "GAME_DATE"
    assert list(result.columns) == ["PTS", "REB", "AST"]
    # non-chart columns (MATCHUP, MIN) are dropped
    assert "MIN" not in result.columns


def test_melt_for_overlay_row_count_and_player_column():
    df = _sample_game_log()
    melted = charts.melt_for_overlay(df, "LeBron James")
    # 2 games * 3 stat cols = 6 rows
    assert len(melted) == 6
    assert set(melted["Player"]) == {"LeBron James"}
    assert set(melted["Stat"]) == {"PTS", "REB", "AST"}
    assert "MATCHUP" not in melted.columns


def test_build_overlay_chart_returns_alt_chart_with_expected_encoding():
    df1 = charts.melt_for_overlay(_sample_game_log(), "LeBron James")
    df2 = charts.melt_for_overlay(_sample_game_log(), "Kevin Durant")
    combined = pd.concat([df1, df2])
    chart = charts.build_overlay_chart(combined)
    assert isinstance(chart, alt.Chart)
    assert chart.encoding.x.shorthand == "GAME_DATE:T"
    assert chart.encoding.color.shorthand == "Player:N"
    assert chart.encoding.strokeDash.shorthand == "Stat:N"


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
