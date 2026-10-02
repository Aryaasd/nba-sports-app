"""Markup for the app's visual system (styles live in ui.css).

Pure functions that return HTML strings for st.html, so layout is unit-testable without a
browser and sports_app.py stays about data and flow. Every value placed into markup is
HTML-escaped: player names come from nba_api, and "Nah'Shon" or "D'Angelo" must not break
an attribute.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent / "ui.css"

# NBA.com's headshot CDN. An unknown ID still returns 200 with a gray silhouette, so a
# card never shows a broken image.
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

# Official (primary, secondary) colors. Where a team's secondary is black or a navy that
# disappears on this dark UI, its next official accent is used instead.
TEAM_COLORS = {
    "ATL": ("#E03A3E", "#C1D32F"),
    "BKN": ("#000000", "#FFFFFF"),
    "BOS": ("#007A33", "#BA9653"),
    "CHA": ("#1D1160", "#00788C"),
    "CHI": ("#CE1141", "#FFFFFF"),
    "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"),
    "DEN": ("#0E2240", "#FEC524"),
    "DET": ("#C8102E", "#1D42BA"),
    "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#C4CED4"),
    "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"),
    "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#F5B112"),
    "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"),
    "MIN": ("#0C2340", "#78BE20"),
    "NOP": ("#0C2340", "#85714D"),
    "NYK": ("#006BB6", "#F58426"),
    "OKC": ("#007AC1", "#EF3B24"),
    "ORL": ("#0077C0", "#C4CED4"),
    "PHI": ("#006BB6", "#ED174C"),
    "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#FFFFFF"),
    "SAC": ("#5A2D81", "#63727A"),
    "SAS": ("#C4CED4", "#000000"),
    "TOR": ("#CE1141", "#A1A1A4"),
    "UTA": ("#4E2A84", "#FFFFFF"),
    "WAS": ("#002B5C", "#E31837"),
}
NEUTRAL_COLORS = ("#2A3747", "#A7B0BC")
# Below this relative luminance a team color is too dark to glow on the card's surface.
_MIN_GLOW_LUMINANCE = 0.03

def global_css() -> str:
    return f"<style>{CSS_PATH.read_text()}</style>"


def headshot_url(player_id: int) -> str:
    return HEADSHOT_URL.format(player_id=int(player_id))


def team_colors(abbreviation: str | None) -> tuple[str, str]:
    return TEAM_COLORS.get(abbreviation or "", NEUTRAL_COLORS)


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of "#RRGGBB", 0 (black) to 1 (white)."""
    channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def glow_color(primary: str, secondary: str) -> str:
    """The team color lit behind a headshot: the primary, unless it's too dark to see."""
    return primary if relative_luminance(primary) >= _MIN_GLOW_LUMINANCE else secondary


def _key(series_color: str) -> str:
    """A dot in the color a player takes in the page's charts."""
    return f'<span class="key" style="background:{escape(series_color)}"></span>'


def brand() -> str:
    return (
        '<div class="brand"><div class="brand-kicker">NBA Sports App</div>'
        '<div class="brand-name">Box score,<br><span>cross-examined</span></div></div>'
    )


def credits(lines: list[str]) -> str:
    return '<div class="credits">' + "<br>".join(escape(line) for line in lines) + "</div>"


def hero(eyebrow: str, title_lines: list[str], blurb: str) -> str:
    title = "<br>".join(escape(line) for line in title_lines)
    return (
        f'<section class="hero">'
        f'<div class="eyebrow">{escape(eyebrow)}</div><h1>{title}</h1><p>{escape(blurb)}</p></section>'
    )


def page_header(eyebrow: str, title: str, blurb: str) -> str:
    return (
        f'<header class="page-head"><div class="eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(blurb)}</p></header>"
    )


def section_header(title: str, note: str | None = None) -> str:
    aside = f'<span class="eyebrow">{escape(note)}</span>' if note else ""
    return f'<div class="section-head"><h2>{escape(title)}</h2>{aside}</div>'


def finding_card(kicker: str, figure: str, body: str, versus: str | None = None) -> str:
    """A headline number from the app's own results. `versus` adds a second figure ("A vs B")."""
    shown = escape(figure)
    if versus is not None:
        shown += f"<small>vs</small>{escape(versus)}"
    return (
        f'<div class="finding"><div class="eyebrow">{escape(kicker)}</div>'
        f'<div class="figure">{shown}</div><p>{escape(body)}</p></div>'
    )


def tile(title: str, body: str) -> str:
    return f'<div class="tile"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>'


def player_card(
    player_id: int,
    name: str,
    *,
    team: str | None = None,
    games: int | None = None,
    averages: dict[str, float] | None = None,
    series_color: str | None = None,
    pill: str | None = None,
    note: str | None = None,
) -> str:
    """Headshot, team stripes, and a season line. `series_color` marks the color this player
    takes in the page's charts; `note` replaces the stat line when there isn't one."""
    primary, secondary = team_colors(team)
    style = f"--team:{primary};--team2:{secondary};--glow:{glow_color(primary, secondary)}"

    key = _key(series_color) if series_color else ""
    meta = " · ".join(
        part for part in [team, f"{games} games" if games is not None else None] if part
    )
    if averages:
        stats = "".join(
            f'<div class="stat"><div class="v">{value:.1f}</div><div class="k">{escape(stat)}</div></div>'
            for stat, value in averages.items()
        )
        line = f'<div class="pc-line">{stats}</div>'
    else:
        line = ""
    extra = f'<div class="pc-note">{escape(note)}</div>' if note else ""
    badge = f'<span class="pill">{escape(pill)}</span>' if pill else ""

    return (
        f'<div class="player-card" style="{style}"><div class="pc-body">'
        f'<div class="eyebrow">{key}{escape(meta)}</div><div class="pc-name">{escape(name)}</div>'
        f"{line}{extra}{badge}</div>"
        f'<img class="pc-shot" src="{headshot_url(player_id)}" alt=""></div>'
    )


def player_label(name: str, series_color: str) -> str:
    """A player's name keyed to his chart color, for sections that discuss both players."""
    return f'<div class="eyebrow">{_key(series_color)}{escape(name)}</div>'


def versus() -> str:
    return '<div class="versus" aria-hidden="true">VS</div>'
