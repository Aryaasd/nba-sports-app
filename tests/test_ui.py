from nba_api.stats.static import teams

import ui


def test_every_team_has_colors():
    # nba_api's team list is bundled data, not a network call.
    abbreviations = {team["abbreviation"] for team in teams.get_teams()}
    assert set(ui.TEAM_COLORS) == abbreviations


def test_unknown_team_gets_neutral_colors():
    assert ui.team_colors(None) == ui.NEUTRAL_COLORS
    assert ui.team_colors("XYZ") == ui.NEUTRAL_COLORS


def test_relative_luminance_endpoints():
    assert ui.relative_luminance("#000000") == 0
    assert abs(ui.relative_luminance("#FFFFFF") - 1) < 1e-9


def test_glow_falls_back_to_secondary_when_primary_is_too_dark():
    assert ui.glow_color(*ui.TEAM_COLORS["LAL"]) == "#552583"
    # Minnesota's navy disappears on the card surface, so its green glows instead.
    assert ui.glow_color(*ui.TEAM_COLORS["MIN"]) == "#78BE20"


def test_headshot_url_uses_player_id():
    assert ui.headshot_url(2544).endswith("/2544.png")


def test_player_card_shows_season_line_and_team_colors():
    card = ui.player_card(
        2544,
        "LeBron James",
        team="LAL",
        games=60,
        averages={"PTS": 20.94, "REB": 6.1, "AST": 7.25},
        series_color="#3987e5",
        pill="Fairly paid · $52.6M",
    )
    assert "LAL · 60 games" in card
    assert ">20.9<" in card and ">7.2<" in card
    assert "--team:#552583" in card
    assert "background:#3987e5" in card
    assert "Fairly paid · $52.6M" in card
    assert ui.headshot_url(2544) in card


def test_player_card_without_stats_shows_note_instead():
    card = ui.player_card(201142, "Kevin Durant", note="Live stats unavailable right now.")
    assert "pc-line" not in card
    assert "Live stats unavailable right now." in card
    assert f"--team:{ui.NEUTRAL_COLORS[0]}" in card


def test_markup_escapes_text():
    card = ui.player_card(1, "D'Angelo <b>Russell</b>", note="a & b")
    assert "<b>" not in card
    assert "D&#x27;Angelo &lt;b&gt;Russell&lt;/b&gt;" in card
    assert "a &amp; b" in card
    assert "&lt;i&gt;" in ui.finding_card("<i>", "$1M", "body")


def test_finding_card_versus_figure():
    card = ui.finding_card("Contract value", "$5.1M", "body", versus="$26.4M")
    assert "$5.1M<small>vs</small>$26.4M" in card


def test_global_css_wraps_stylesheet():
    css = ui.global_css()
    assert css.startswith("<style>") and css.endswith("</style>")
    assert "--maple" in css
