"""
Premier League managerial changes, 2019-20 to the current season -> data/manager_changes.csv.

    python -m scripts.data.fetch_manager_changes [--first 2019] [--last 2026] [--out data/manager_changes.csv]

Source: the "Managerial changes" table of each "YYYY-YY Premier League" article on English
Wikipedia (text available under CC BY-SA 4.0), read through the MediaWiki API as wikitext.
One row per change: team (the graph's FPL short name), outgoing manager, manner of departure,
date of vacancy, position, incoming manager, whether the incoming one is a caretaker/interim,
and date of appointment. Pre-season changes are listed under the season they precede.

Used by the backtest harness (scripts/analysis/ml_backtest.py, "mgr" variants) only; the points
model doesn't use it, so nothing refreshes it automatically. Promoted clubs' changes before
promotion aren't in these tables.
"""
import argparse
import re
from typing import Dict, List

import httpx
import pandas as pd

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "PitchMe-FPL/1.0 (https://github.com/youssef20031/ACL_M3)"}
OUT = "data/manager_changes.csv"

TEAMS = {
    "arsenal": "Arsenal", "aston villa": "Aston Villa", "bournemouth": "Bournemouth", "brentford": "Brentford",
    "brighton": "Brighton", "burnley": "Burnley", "chelsea": "Chelsea", "crystal palace": "Crystal Palace",
    "everton": "Everton", "fulham": "Fulham", "ipswich": "Ipswich", "leeds": "Leeds", "leicester": "Leicester",
    "liverpool": "Liverpool", "luton": "Luton", "manchester city": "Man City", "manchester united": "Man Utd",
    "newcastle": "Newcastle", "norwich": "Norwich", "nottingham": "Nott'm Forest", "sheffield united": "Sheffield Utd",
    "southampton": "Southampton", "tottenham": "Spurs", "sunderland": "Sunderland", "watford": "Watford",
    "west bromwich": "West Brom", "west brom": "West Brom", "west ham": "West Ham", "wolverhampton": "Wolves",
    "wolves": "Wolves", "hull": "Hull City", "coventry": "Coventry City",
}

# Cell attributes before the content, e.g. `rowspan="2"|`, `data-sort-value="Slot, Arne"|`, `nowrap|`;
# the wikitext isn't always tidy (`rowspan=2|`, `rowspan="2|16th`, `rowspan=|`)
ATTR = re.compile(r'^\s*((?:\s*(?:(?:rowspan|colspan|data-sort-value|style|align|class|scope)\s*=\s*'
                  r'(?:"[^"|]*"?|[^\s|"]*)|nowrap))+)\s*\|(?!\|)(.*)$', re.S)


def _strip_refs(text: str) -> str:
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def _clean(cell: str) -> str:
    cell = re.sub(r"\{\{#invoke:flag\|[^}]*\}\}", "", cell)
    cell = re.sub(r"\{\{[Ff]lagicon\|[^}]*\}\}", "", cell)
    cell = re.sub(r"\{\{nowrap\|([^}]*)\}\}", r"\1", cell)
    cell = re.sub(r"\{\{sortname\|([^|}]*)\|([^|}]*)(\|[^}]*)?\}\}", r"\1 \2", cell)
    cell = re.sub(r"\{\{[^}]*\}\}", "", cell)
    cell = re.sub(r"\[\[([^|\]]*)\|([^\]]*)\]\]", r"\2", cell)
    cell = re.sub(r"\[\[([^\]]*)\]\]", r"\1", cell)
    cell = re.sub(r"<[^>]+>", " ", cell).replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", cell).strip().lstrip("| ").strip()


def parse_table(wikitext: str) -> List[Dict[str, str]]:
    """Rows of the first wikitable in `wikitext` as {header: text}, with rowspans expanded."""
    text = _strip_refs(wikitext)
    start = text.index("{|")
    body = text[start:text.index("\n|}", start)]
    header, rows, pending = None, [], {}
    for chunk in re.split(r"\n\|-[^\n]*", body):
        lines = [line for line in chunk.split("\n") if line.strip() and not line.startswith("{|")]
        names = [_clean(h) for line in lines if line.startswith("!") for h in line[1:].split("!!")]
        if names:
            header = names
            continue
        cells = []
        for line in lines:
            if line.startswith("|"):
                cells += line[1:].split("||")
            elif cells:
                cells[-1] += " " + line
        if not cells or header is None:
            continue
        parsed = []
        for cell in cells:
            m = ATTR.match(cell)
            attrs, content = (m.group(1), m.group(2)) if m else ("", cell)
            span = re.search(r'rowspan\s*=\s*"?(\d+)', attrs)
            parsed.append((int(span.group(1)) if span else 1, _clean(content)))
        row, it = [], iter(parsed)
        for col in range(len(header)):
            if pending.get(col, (0, None))[0] > 0:
                left, value = pending[col]
                pending[col] = (left - 1, value)
                row.append(value)
                continue
            span, value = next(it, (1, None))
            row.append(value)
            if span > 1:
                pending[col] = (span - 1, value)
        rows.append(dict(zip(header, row)))
    return rows


def _team(name: str) -> str:
    key = (name or "").lower()
    for fragment, team in TEAMS.items():
        if fragment in key:
            return team
    raise ValueError(f"unknown team {name!r}")


def _date(text: str):
    d = pd.to_datetime(re.sub(r"\(.*?\)", "", text or "").strip(), dayfirst=True, errors="coerce")
    return None if pd.isna(d) else d.date().isoformat()


def section_wikitext(client: httpx.Client, page: str, title: str) -> str:
    sections = client.get(API, params={"action": "parse", "page": page, "prop": "sections", "format": "json",
                                       "redirects": 1}).json()["parse"]["sections"]
    index = next(s["index"] for s in sections if s["line"] == title)
    return client.get(API, params={"action": "parse", "page": page, "prop": "wikitext", "section": index,
                                   "format": "json", "redirects": 1}).json()["parse"]["wikitext"]["*"]


def fetch(first: int, last: int) -> pd.DataFrame:
    rows = []
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for year in range(first, last + 1):
            season = f"{year}-{str(year + 1)[2:]}"
            text = section_wikitext(client, f"{year}–{str(year + 1)[2:]} Premier League", "Managerial changes")
            for r in parse_table(text):
                incoming = r.get("Incoming manager") or ""
                rows.append({
                    "season": season, "team": _team(r.get("Team")), "outgoing": r.get("Outgoing manager"),
                    "manner": r.get("Manner of departure"), "vacancy": _date(r.get("Date of vacancy")),
                    "position": r.get("Position in the table") or r.get("Position in table"),
                    "incoming": re.sub(r"\s*\((caretaker|interim)\)", "", incoming, flags=re.I).strip(),
                    "incoming_interim": bool(re.search(r"caretaker|interim", incoming, re.I)),
                    "appointed": _date(r.get("Date of appointment")),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Premier League managerial changes from Wikipedia")
    parser.add_argument("--first", type=int, default=2019, help="first season's start year")
    parser.add_argument("--last", type=int, default=pd.Timestamp.now().year - (pd.Timestamp.now().month < 7))
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    df = fetch(args.first, args.last)
    missing = df[df["vacancy"].isna() | df["appointed"].isna()]
    if len(missing):
        print(f"warning: {len(missing)} rows without a date:\n{missing.to_string()}")
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"{args.out}: {len(df)} changes, {df.groupby('season').size().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
