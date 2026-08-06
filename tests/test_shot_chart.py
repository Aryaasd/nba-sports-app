import pandas as pd
import pytest

from insights import shot_chart


def test_bin_shots_aggregates_attempts_makes_and_fg_pct():
    df = pd.DataFrame(
        {
            "LOC_X": [5, 8, 45, 100],
            "LOC_Y": [5, 10, 45, 200],
            "SHOT_MADE_FLAG": [1, 0, 1, 1],
        }
    )
    result = shot_chart.bin_shots(df, bin_size=40)

    bin00 = result[(result["x_bin"] == 0) & (result["y_bin"] == 0)].iloc[0]
    assert bin00["attempts"] == 2
    assert bin00["makes"] == 1
    assert bin00["fg_pct"] == pytest.approx(0.5)

    bin_other = result[(result["x_bin"] == 40) & (result["y_bin"] == 40)].iloc[0]
    assert bin_other["attempts"] == 1
    assert bin_other["fg_pct"] == pytest.approx(1.0)


def test_bin_shots_negative_coordinates_floor_correctly():
    # -47 falls in the [-80, -40) bin under floor-division binning.
    df = pd.DataFrame({"LOC_X": [-47], "LOC_Y": [10], "SHOT_MADE_FLAG": [0]})
    result = shot_chart.bin_shots(df, bin_size=40)
    row = result.iloc[0]
    assert row["x_bin"] == -80
    assert row["x0"] == -80
    assert row["x1"] == -40


def test_bin_shots_bin_edges():
    df = pd.DataFrame({"LOC_X": [10], "LOC_Y": [20], "SHOT_MADE_FLAG": [1]})
    result = shot_chart.bin_shots(df, bin_size=40)
    row = result.iloc[0]
    assert (row["x0"], row["x1"]) == (0, 40)
    assert (row["y0"], row["y1"]) == (0, 40)


def test_court_lines_df_has_expected_shapes_and_path_order():
    df = shot_chart.court_lines_df()
    line_ids = set(df["line_id"])
    assert {"hoop", "three_point_arc", "restricted_area", "paint", "court_boundary"} <= line_ids
    # seq gives each line_id's own points a stable path order, starting at 0
    hoop = df[df["line_id"] == "hoop"].sort_values("seq")
    assert list(hoop["seq"]) == list(range(len(hoop)))


@pytest.mark.parametrize(
    "x0, y0, expected",
    [
        (0, 0, "Restricted Area"),  # center (20, 20), distance ~28.3 < 40
        (100, 100, "Mid-Range"),  # center (120, 120), distance ~169.7
        (200, 200, "Three-Point"),  # center (220, 220), distance ~311.1 >= 237.5
    ],
)
def test_label_zone(x0, y0, expected):
    assert shot_chart.label_zone(x0, y0, bin_size=40) == expected
