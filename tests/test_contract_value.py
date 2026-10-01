import numpy as np
import pandas as pd
import pytest

from insights import contract_value as cv


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Nikola Jokić", "nikola jokic"),
        ("Jaren Jackson Jr.", "jaren jackson"),
        ("A.J. Lawson", "aj lawson"),
        ("Shai Gilgeous-Alexander", "shai gilgeous alexander"),
        ("Gary Trent Jr", "gary trent"),
        ("Kelly Oubre Jr.", "kelly oubre"),
        ("D'Angelo Russell", "dangelo russell"),
        ("Egor Dëmin", "egor demin"),
        ("  Trey   Murphy III ", "trey murphy"),
    ],
)
def test_normalize_name(raw, expected):
    assert cv.normalize_name(raw) == expected


def _players():
    names = ["Nikola Jokić", "Jaren Jackson Jr.", "Herbert Jones"]
    return pd.DataFrame({"PLAYER_ID": [1, 2, 3], "PLAYER_NAME": names})


def test_join_salaries_matches_across_spellings_and_reports_unmatched():
    salaries = pd.DataFrame(
        {"PLAYER_NAME": ["Nikola Jokic", "Jaren Jackson", "Nobody Real"], "SALARY": [55e6, 35e6, 1e6]}
    )
    joined, unmatched = cv.join_salaries(_players(), salaries)
    assert dict(zip(joined["PLAYER_ID"], joined["SALARY"])) == {1: 55e6, 2: 35e6}
    assert unmatched == ["Nobody Real"]


def test_join_salaries_aliases_resolve_nicknames():
    salaries = pd.DataFrame({"PLAYER_NAME": ["Herb Jones"], "SALARY": [13e6]})
    joined, unmatched = cv.join_salaries(_players(), salaries, aliases={"Herb Jones": "Herbert Jones"})
    assert joined["PLAYER_ID"].tolist() == [3]
    assert unmatched == []


def test_join_salaries_refuses_duplicate_salary_rows():
    salaries = pd.DataFrame({"PLAYER_NAME": ["Nikola Jokic", "Nikola Jokić"], "SALARY": [55e6, 1e6]})
    with pytest.raises(ValueError, match="more than once"):
        cv.join_salaries(_players(), salaries)


def test_qualifies_needs_both_games_and_minutes():
    assert cv.qualifies(cv.MIN_GAMES, cv.MIN_MINUTES_PER_GAME)
    assert not cv.qualifies(cv.MIN_GAMES - 1, 30)
    assert not cv.qualifies(70, cv.MIN_MINUTES_PER_GAME - 0.1)


def test_verdict_band_uses_typical_log_error():
    typical = 0.4  # model's typical miss in log-salary: about 1.5x either way
    assert cv.verdict(14e6, 10e6, typical, qualified=True)["label"] == cv.FAIR  # log(1.4) = 0.34
    assert cv.verdict(15e6, 10e6, typical, qualified=True)["label"] == cv.OVERPAID  # log(1.5) = 0.405
    assert cv.verdict(20e6, 10e6, typical, qualified=True)["label"] == cv.OVERPAID
    assert cv.verdict(5e6, 10e6, typical, qualified=True)["label"] == cv.UNDERPAID
    assert np.isclose(cv.verdict(20e6, 10e6, typical, qualified=True)["ratio"], 2.0)


def test_non_market_contracts_are_never_called_over_or_under():
    assert cv.verdict(3e6, 30e6, 0.4, qualified=True, rookie_scale=True)["label"] == cv.ROOKIE_SCALE
    assert cv.verdict(2.3e6, 26e6, 0.4, qualified=True, minimum_deal=True)["label"] == cv.MINIMUM_DEAL
    assert cv.verdict(46e6, 78e6, 0.4, qualified=True, near_max=True)["label"] == cv.MAX_DEAL
    assert cv.verdict(3e6, 30e6, 0.4, qualified=False) == {"label": cv.NOT_ENOUGH_MINUTES, "ratio": None}


def test_max_player_paid_well_above_production_can_still_be_called_overpaid():
    # The ceiling only explains a gap in one direction: it can't make a max deal look too big.
    assert cv.verdict(50e6, 15e6, 0.4, qualified=True, near_max=True)["label"] == cv.OVERPAID


def test_partial_deal_gets_no_verdict_even_with_plenty_of_minutes():
    result = cv.verdict(131_970, 8e6, 0.4, qualified=True, partial_deal=True)
    assert result == {"label": cv.PARTIAL_DEAL, "ratio": None}
    assert "fraction of a season's pay" in cv.describe_verdict(result, 8e6)


def test_join_salaries_carries_extra_columns_across():
    salaries = pd.DataFrame({"PLAYER_NAME": ["Nikola Jokic"], "SALARY": [55e6], "ROOKIE_SCALE": [False]})
    joined, _ = cv.join_salaries(_players(), salaries)
    assert joined.loc[0, "ROOKIE_SCALE"] == False  # noqa: E712 -- numpy bool


def test_describe_max_deal_names_the_ceiling_only_when_production_prices_above_it():
    above_ceiling = cv.verdict(46e6, 78e6, 0.4, qualified=True, near_max=True)
    sentence = cv.describe_verdict(above_ceiling, 78e6)
    assert "salary cap, not the market" in sentence and "$54.1M ceiling" in sentence
    below_ceiling = cv.verdict(40e6, 48e6, 0.4, qualified=True, near_max=True)
    assert "ceiling" not in cv.describe_verdict(below_ceiling, 48e6)


def test_describe_underpaid_and_reference_values():
    assert "2.4x his salary" in cv.describe_verdict(cv.verdict(10e6, 24e6, 0.4, qualified=True), 24e6)
    rookie = cv.verdict(3e6, 30e6, 0.4, qualified=True, rookie_scale=True)
    assert "$30.0M" in cv.describe_verdict(rookie, 30e6)
    minimum = cv.describe_verdict(cv.verdict(2.3e6, 26e6, 0.4, qualified=True, minimum_deal=True), 26e6)
    assert "price floor" in minimum and "$26.0M" in minimum


def test_load_contract_values_missing_file_is_none(tmp_path):
    assert cv.load_contract_values(tmp_path / "missing.parquet") is None
