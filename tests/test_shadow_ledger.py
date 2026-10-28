"""Tests for scripts/analysis/shadow_ledger.py: running totals, the paired bootstrap and the table."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import shadow_ledger as sl


def row(gw, live, shadow, status="finished", live_model="v4", shadow_model="v3"):
    return {"gameweek": gw, "status": status, "live_model": live_model, "shadow_model": shadow_model,
            "live_points": live, "shadow_points": shadow, "difference": live - shadow,
            "live_predicted": 50.0, "shadow_predicted": 48.0, "shared_players": 9, "fpl_average": 52}


def test_running_total_counts_finished_gameweeks_of_two_models_only():
    rows = sl.ledger([row(9, 60, 50), row(10, 40, 45), row(11, 70, 70, live_model="v3"),   # same model
                      row(12, 55, 50), row(13, 30, 20, status="live")])
    assert [r["running"] for r in rows] == [10, 5, None, 10, None]
    assert [sl.counted(r) for r in rows] == [True, True, False, True, False]


def test_bootstrap_ci_of_the_mean_difference():
    s = sl.bootstrap_ci([10, -5, 5])
    assert (s["gameweeks"], s["total"], s["mean"]) == (3, 10.0, pytest.approx(10 / 3))
    assert s["lo"] <= s["mean"] <= s["hi"] and -5 <= s["lo"] and s["hi"] <= 10
    assert sl.bootstrap_ci([10, -5, 5]) == s                       # seeded: the same every run
    constant = sl.bootstrap_ci([4, 4, 4, 4])
    assert constant["lo"] == constant["hi"] == 4.0 and constant["p_live_ahead"] == 1.0
    assert sl.bootstrap_ci([-3, -1])["p_live_ahead"] == 0.0
    one = sl.bootstrap_ci([7])
    assert one["mean"] == 7.0 and one["lo"] is None and one["hi"] is None
    assert sl.bootstrap_ci([])["mean"] is None


def test_bootstrap_ci_matches_the_spread_of_the_mean():
    rng = np.random.default_rng(1)
    diffs = rng.normal(2.0, 10.0, size=60)
    s = sl.bootstrap_ci(diffs)
    half = (s["hi"] - s["lo"]) / 2
    assert half == pytest.approx(1.96 * diffs.std() / np.sqrt(len(diffs)), rel=0.1)


def test_table_lists_every_gameweek_and_the_summary():
    rows = sl.ledger([row(9, 60, 50), row(10, 40, 45), row(13, 30, 20, status="live")])
    text = sl.format_table(rows, sl.bootstrap_ci(r["difference"] for r in rows if sl.counted(r)))
    lines = text.splitlines()
    assert lines[2].split()[:6] == ["9", "finished", "60", "50", "+10", "+10"]
    assert "(provisional)" in lines[4]
    assert "2 finished gameweeks: live - shadow = +5 points in total, +2.50 per gameweek" in text
    assert "No finished gameweek" in sl.format_table([], sl.bootstrap_ci([]))


def test_main_reads_the_saved_picks(tmp_path, monkeypatch, capsys):
    """End to end on a live dir with one finished gameweek: both picks scored like FPL."""
    from services import season_service as ss

    picks_dir = tmp_path / "model_picks" / "2026-27"
    (picks_dir / "shadow").mkdir(parents=True)

    def pick(captain, model):
        players = [{"element": k, "name": f"P{k}", "position": pos, "team": "T", "cost": 5.0,
                    "predicted_points": 2.0, "role": "starter" if k <= 11 else "bench",
                    "captain": k == captain, "vice_captain": k == 2, "fixtures": [1], "opponents": []}
                   for k, pos in enumerate(["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
                                           + ["GK", "DEF", "MID", "FWD"], start=1)]
        return {"season": "2026-27", "gameweek": 9, "deadline": "2026-10-24T10:00:00Z", "model": model,
                "players": players}

    (picks_dir / "gw09.json").write_text(json.dumps(pick(10, "v4")))                # captain scored 8
    (picks_dir / "shadow" / "gw09.json").write_text(json.dumps(pick(3, "v3")))      # captain scored 2
    pd.DataFrame([{"element": k, "GW": 9, "minutes": 90, "total_points": 8 if k == 10 else 2}
                  for k in range(1, 16)]).to_csv(tmp_path / "FPL_2026_2027.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({
        "season": "2026-27", "files": {"season": "FPL_2026_2027.csv"}, "finished_fixtures": [1],
        "events": [{"id": 9, "finished": True, "average_entry_score": 50, "highest_score": 120}]}))
    monkeypatch.setenv("LIVE_DATA_DIR", "unused")
    ss.clear_cache()
    try:
        assert sl.main(["--live-dir", str(tmp_path), "--season", "2026-27", "--json"]) == 0
    finally:
        ss.clear_cache()
    out = json.loads(capsys.readouterr().out)
    assert os.environ["LIVE_DATA_DIR"] == str(tmp_path)
    r = out["rows"][0]
    # live: 8 x2 + ten others 2 each = 36; shadow: 2 x2 + 8 + nine others 2 each = 30
    assert (r["gameweek"], r["live_points"], r["shadow_points"], r["difference"], r["running"]) == (9, 36, 30, 6, 6)
    assert out["summary"]["gameweeks"] == 1 and out["summary"]["total"] == 6.0
