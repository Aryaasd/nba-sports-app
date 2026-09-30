import pandas as pd
import pytest

from insights import defense_tiers

TEAM_ID_TO_ABBR = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612747: "LAL",
    1610612748: "MIA",
}


@pytest.mark.parametrize(
    "matchup, expected",
    [
        ("LAL vs. HOU", "HOU"),
        ("LAL @ DAL", "DAL"),
        ("BOS vs. MIA", "MIA"),
    ],
)
def test_extract_opponent_abbreviation(matchup, expected):
    assert defense_tiers.extract_opponent_abbreviation(matchup) == expected


@pytest.mark.parametrize(
    "matchup, expected",
    [
        ("LAL vs. HOU", "LAL"),
        ("LAL @ DAL", "LAL"),
        ("BOS vs. MIA", "BOS"),
    ],
)
def test_extract_own_team_abbreviation(matchup, expected):
    assert defense_tiers.extract_own_team_abbreviation(matchup) == expected


@pytest.mark.parametrize(
    "matchup, expected",
    [
        ("OKC vs. IND", True),
        ("IND @ OKC", False),
        ("LAL vs. HOU", True),
        ("LAL @ DAL", False),
    ],
)
def test_is_home_game(matchup, expected):
    assert defense_tiers.is_home_game(matchup) is expected


def _team_advanced_df():
    # Median of [118.4, 108.4, 112.0, 122.0] is 115.2 -> BOS/LAL Elite, ATL/MIA Weak.
    return pd.DataFrame(
        {
            "TEAM_ID": [1610612737, 1610612738, 1610612747, 1610612748],
            "DEF_RATING": [118.4, 108.4, 112.0, 122.0],
        }
    )


def test_bucket_teams_by_defense_median_split():
    tiers = defense_tiers.bucket_teams_by_defense(_team_advanced_df(), TEAM_ID_TO_ABBR)
    assert tiers["BOS"] == defense_tiers.ELITE_LABEL
    assert tiers["LAL"] == defense_tiers.ELITE_LABEL
    assert tiers["ATL"] == defense_tiers.WEAK_LABEL
    assert tiers["MIA"] == defense_tiers.WEAK_LABEL


def test_compare_performance_by_tier_grouped_means_and_counts():
    team_tier_map = {"BOS": "Elite Defense", "MIA": "Weak Defense"}
    game_log = pd.DataFrame(
        {
            "MATCHUP": ["LAL vs. BOS", "LAL @ BOS", "LAL vs. MIA"],
            "PTS": [20, 24, 30],
            "REB": [5, 7, 9],
            "AST": [4, 6, 8],
        }
    )
    result = defense_tiers.compare_performance_by_tier(game_log, team_tier_map)
    assert result.loc["Elite Defense", "PTS"] == pytest.approx((20 + 24) / 2)
    assert result.loc["Elite Defense", "Games"] == 2
    assert result.loc["Weak Defense", "PTS"] == pytest.approx(30)
    assert result.loc["Weak Defense", "Games"] == 1


def test_compare_performance_by_tier_unknown_opponents_dropped():
    team_tier_map = {"BOS": "Elite Defense"}
    game_log = pd.DataFrame(
        {"MATCHUP": ["LAL vs. XXX"], "PTS": [20], "REB": [5], "AST": [4]}
    )
    result = defense_tiers.compare_performance_by_tier(game_log, team_tier_map)
    assert result.empty
