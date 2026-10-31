"""Predicted players match by match (services/fixture_picks.py) and its assistant card."""
from services import answer_cards as ac
from services import fixture_picks as fp


def row(name, team, opponent, home, points, pos="MID", kickoff="2026-11-07T15:00:00Z", **kw):
    return {"name": name, "team": team, "opponent": opponent, "is_home": home, "predicted_points": points,
            "position": pos, "kickoff_time": kickoff, "gameweek": 11, "status": "a", **kw}


ROWS = [
    row("Haaland", "Man City", "Chelsea", True, 8.1, "FWD"),
    row("Palmer", "Chelsea", "Man City", False, 6.0),
    row("Foden", "Man City", "Chelsea", True, 5.2),
    row("Enzo", "Chelsea", "Man City", False, 4.0),
    row("Injured", "Chelsea", "Man City", False, 0.0, status="i"),
    row("Saka", "Arsenal", "Spurs & Leeds", False, 11.0, kickoff="2026-11-07T12:30:00Z"),
    row("Son", "Spurs", "Arsenal", True, 4.5, kickoff="2026-11-07T12:30:00Z"),
    row("Nobody", "Burnley", "No fixture", False, 0.0),
]


def test_detects_per_match_questions():
    for q in ["Players expected to get the most points for each match in the next GW",
              "who will score the most points in every fixture this gameweek",
              "top 2 predicted players per game next gw", "Best player in each match next gameweek",
              "match by match predictions"]:
        assert fp.is_per_fixture_question(q), q
    for q in ["results of every match last season", "Who should I captain?", "Haaland predicted points",
              "how many goals in each game did Arsenal score in 2023-24"]:
        assert not fp.is_per_fixture_question(q), q


def test_count():
    assert fp.requested_count("top 2 players in each match next gw") == 2
    assert fp.requested_count("the best three players per game next gw") == 3
    assert fp.requested_count("each match next gw") == fp.DEFAULT_PER_FIXTURE
    assert fp.requested_count("top 9 per fixture next gw") == fp.MAX_PER_FIXTURE


def test_groups_both_sides_by_kickoff():
    groups = fp.top_by_fixture(ROWS, 2)
    assert [(g["home"], g["away"]) for g in groups] == [("Spurs", "Arsenal"), ("Man City", "Chelsea")]
    assert [p["name"] for p in groups[1]["players"]] == ["Haaland", "Palmer"]
    saka = groups[0]["players"][0]
    assert saka["name"] == "Saka" and saka["double"]              # listed once, under his first match
    assert all(p["name"] != "Injured" for g in groups for p in g["players"])


def test_context_lists_every_match():
    text = fp.context(fp.top_by_fixture(ROWS, 3), 3)
    assert "Spurs v Arsenal (Sat 07 Nov 12:30 UTC)" in text and "Man City v Chelsea" in text
    assert "1. Haaland (FWD, Man City): 8.1 pts" in text
    assert "double gameweek" in text
    assert fp.context([], 3).startswith("There are no predictions")


def test_card_and_plan():
    card = fp.card(fp.top_by_fixture(ROWS, 3), 3, player_ref=ac.player_ref, photo=lambda n, t, c: None, codes={})
    assert card["type"] == "fixture_picks" and card["gameweek"] == 11 and card["per_fixture"] == 3
    assert card["fixtures"][1]["players"][0] == {
        "player": ac.player_ref("Haaland", "Man City", "FWD", None, "a", None, None), "points": 8.1, "double": False}
    assert ("Saka stats this season", "player_stats") in card["_follow_ups"]
    assert fp.card([], 3, player_ref=ac.player_ref, photo=lambda n, t, c: None, codes={}) is None
