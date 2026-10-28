"""Timing and join guarantees of ml/snapshots.py, FPL's availability and set-piece snapshots
before each deadline (synthetic frames, no files)."""
import numpy as np
import pandas as pd
import pytest

from ml import snapshots as sn

S = "2030-31"
# Deadlines are each gameweek's first kickoff minus 90 minutes:
# GW1 2030-08-10 12:30, GW2 2030-08-17 12:30 (a double: second fixture 2030-08-20), GW3 2030-08-24 12:30
KICKOFFS = {1: ["2030-08-10T14:00:00Z"], 2: ["2030-08-17T14:00:00Z", "2030-08-20T19:00:00Z"],
            3: ["2030-08-24T14:00:00Z"]}


def _rows(elements=(1, 2)):
    rows = []
    for gw, kicks in KICKOFFS.items():
        for k, kickoff in enumerate(kicks):
            for e in elements:
                rows.append({"season": S, "GW": gw, "fixture": 10 * gw + k, "element": float(e),
                             "kickoff_time": pd.Timestamp(kickoff), "minutes": 90.0})
    return pd.DataFrame(rows)


def _snap(at, sha, players):
    """players: {element: (status, chance, news, penalties, corners, free kicks)}"""
    return [{"season": S, "captured_at": pd.Timestamp(at), "sha": sha, "element": e, "code": 1000 + e,
             "status": st, "chance_of_playing_next_round": ch, "news_flag": nw, "penalties_order": pen,
             "corners_and_indirect_freekicks_order": cor, "direct_freekicks_order": fk}
            for e, (st, ch, nw, pen, cor, fk) in players.items()]


def _snaps(*parts):
    df = pd.DataFrame([r for p in parts for r in p])
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


A = ("a", np.nan, 0, np.nan, np.nan, np.nan)


def test_deadlines_are_the_first_kickoff_minus_90_minutes():
    d = sn.gameweek_deadlines(_rows()).set_index("GW")
    assert d.loc[1, "deadline"] == pd.Timestamp("2030-08-10T12:30:00Z")
    assert d.loc[2, "deadline"] == pd.Timestamp("2030-08-17T12:30:00Z")   # the double's first fixture
    assert pd.isna(d.loc[1, "prev_deadline"])
    assert d.loc[3, "prev_deadline"] == d.loc[2, "deadline"]


def test_the_latest_snapshot_between_the_previous_deadline_and_this_one_wins():
    snaps = _snaps(_snap("2030-08-05T10:00:00Z", "pre", {1: ("d", 50, 1, 1, np.nan, np.nan), 2: A}),
                   _snap("2030-08-12T10:00:00Z", "early", {1: ("i", 0, 1, np.nan, np.nan, np.nan), 2: A}),
                   _snap("2030-08-15T10:00:00Z", "late", {1: ("d", 75, 1, 2, 1, np.nan), 2: A}))
    win = sn.snapshot_windows(sn.gameweek_deadlines(_rows()), snaps).set_index("GW")
    assert win.loc[1, "sha"] == "pre"          # the first gameweek has no lower bound
    assert win.loc[2, "sha"] == "late"         # both are in GW2's window; the later one wins
    f = sn.snapshot_features(_rows(), snaps)
    rows = _rows()
    gw2 = f[(rows["GW"] == 2) & (rows["element"] == 1)]
    assert (gw2["snap_chance"] == 75).all() and (gw2["snap_doubtful"] == 1).all()
    assert (gw2["sp_pen1"] == 0).all() and (gw2["sp_pen"] == 1).all() and (gw2["sp_corners1"] == 1).all()
    assert gw2["snap_age_days"].iloc[0] == pytest.approx((pd.Timestamp("2030-08-17T12:30:00Z")
                                                          - pd.Timestamp("2030-08-15T10:00:00Z")).total_seconds() / 86400)


def test_a_snapshot_from_after_the_deadline_is_never_used():
    # captured at GW1's deadline exactly, and between GW1's kickoff and the end of the gameweek
    snaps = _snaps(_snap("2030-08-10T12:30:00Z", "at", {1: ("i", 0, 1, np.nan, np.nan, np.nan), 2: A}),
                   _snap("2030-08-10T15:00:00Z", "during", {1: ("s", 0, 1, np.nan, np.nan, np.nan), 2: A}))
    rows = _rows()
    f = sn.snapshot_features(rows, snaps)
    assert f.loc[rows["GW"] == 1, sn.FEATURES].isna().all().all()
    assert (f.loc[rows["GW"] == 1, sn.COP] == 1).all()
    # ... but "during" (after GW1's deadline, before GW2's) is GW2's
    assert sn.snapshot_windows(sn.gameweek_deadlines(rows), snaps).set_index("GW").loc[2, "sha"] == "during"


def test_a_snapshot_from_before_the_previous_deadline_is_never_used():
    # The only snapshot before GW3's deadline predates GW2's: its "next round" is GW2, not GW3
    snaps = _snaps(_snap("2030-08-12T10:00:00Z", "old", {1: ("i", 0, 1, np.nan, np.nan, np.nan), 2: A}))
    rows = _rows()
    win = sn.snapshot_windows(sn.gameweek_deadlines(rows), snaps).set_index("GW")
    assert win.loc[2, "sha"] == "old"
    assert pd.isna(win.loc[3, "sha"]) and bool(win.loc[3, "stale"])
    f = sn.snapshot_features(rows, snaps)
    assert f.loc[rows["GW"] == 3, sn.FEATURES].isna().all().all()
    assert (f.loc[rows["GW"] == 3, sn.COP] == 1).all()


def test_a_snapshot_never_crosses_seasons():
    # The previous season's file was committed after this season's pre-season snapshot, and its
    # element 1 is another player: GW1 (no lower bound) still uses only its own season's files
    prev = [dict(r, season="2029-30") for r in _snap("2030-08-09T10:00:00Z", "old", {1: ("u", 0, 1, 1, 1, 1), 2: A})]
    rows = _rows()
    snaps = _snaps(_snap("2030-08-01T10:00:00Z", "pre", {1: A, 2: A}), prev)
    win = sn.snapshot_windows(sn.gameweek_deadlines(rows), snaps).set_index("GW")
    assert win.loc[1, "sha"] == "pre" and not win.loc[1, "stale"]
    gw1 = sn.snapshot_features(rows, snaps)[rows["GW"] == 1]
    assert (gw1["snap_unavailable"] == 0).all() and (gw1[sn.COP] == 1).all()
    assert gw1["sp_pen1"].isna().all()   # "pre" has no set-piece orders at all
    # with only the other season's file, GW1 has no snapshot
    f = sn.snapshot_features(rows, _snaps(prev))
    assert f[sn.FEATURES].isna().all().all() and (f[sn.COP] == 1).all()


def test_both_fixtures_of_a_double_gameweek_share_the_snapshot():
    # one before GW2's deadline, one between the double's two kickoffs (after the deadline)
    snaps = _snaps(_snap("2030-08-16T10:00:00Z", "before", {1: ("d", 25, 1, np.nan, np.nan, np.nan), 2: A}),
                   _snap("2030-08-18T10:00:00Z", "between", {1: ("i", 0, 1, np.nan, np.nan, np.nan), 2: A}))
    rows = _rows()
    f = sn.snapshot_features(rows, snaps)
    gw2 = f[(rows["GW"] == 2) & (rows["element"] == 1)]
    assert len(gw2) == 2
    assert (gw2["snap_chance"] == 25).all() and (gw2[sn.COP] == 0.25).all()
    assert gw2["snap_age_days"].nunique() == 1


def test_the_join_never_duplicates_rows():
    one = {1: ("d", 50, 1, 1, np.nan, np.nan), 2: A}
    snaps = _snaps(_snap("2030-08-16T10:00:00Z", "s1", one), _snap("2030-08-16T10:00:00Z", "s1", one))
    rows = _rows(elements=(1, 2, 3))
    rows.loc[rows["element"] == 3, "element"] = np.nan    # a row without an element
    rows.index = rows.index * 7 + 3                        # any index is kept
    f = sn.snapshot_features(rows, snaps)
    assert len(f) == len(rows) and f.index.equals(rows.index)
    assert f.loc[rows["element"].isna(), sn.FEATURES].isna().all().all()


def test_features_from_the_status_chance_and_set_piece_orders():
    snaps = _snaps(_snap("2030-08-16T10:00:00Z", "s", {1: A, 2: ("u", 0, 1, 1, 1, 1), 3: ("a", 100, 0, 2, 2, 2),
                                                        4: ("s", 0, 1, np.nan, np.nan, np.nan)}))
    rows = _rows(elements=(1, 2, 3, 4, 5))
    f = sn.snapshot_features(rows, snaps)
    gw2 = f[rows["GW"] == 2].groupby(rows["element"]).first()
    assert gw2.loc[1, "snap_chance"] == 100 and gw2.loc[1, "snap_unavailable"] == 0 and gw2.loc[1, "snap_news"] == 0
    assert gw2.loc[2, "snap_unavailable"] == 1 and gw2.loc[2, "snap_chance"] == 0 and gw2.loc[2, sn.COP] == 0
    assert gw2.loc[2, ["sp_pen1", "sp_pen", "sp_corners1", "sp_fk1"]].tolist() == [1, 1, 1, 1]
    assert gw2.loc[3, ["sp_pen1", "sp_pen", "sp_corners1", "sp_fk1"]].tolist() == [0, 1, 0, 0]
    assert gw2.loc[4, "snap_unavailable"] == 1 and gw2.loc[4, "snap_doubtful"] == 0
    assert gw2.loc[5, sn.FEATURES].isna().all() and gw2.loc[5, sn.COP] == 1    # not in the snapshot


def test_set_pieces_are_unknown_in_a_snapshot_without_any_orders():
    snaps = _snaps(_snap("2030-08-16T10:00:00Z", "s", {1: A, 2: ("d", 50, 1, np.nan, np.nan, np.nan)}))
    f = sn.snapshot_features(_rows(), snaps)
    gw2 = f[_rows()["GW"] == 2]
    assert gw2[sn.SP_FEATURES].isna().all().all()
    assert gw2["snap_chance"].notna().all()


def test_no_snapshots_leaves_nan_and_factor_one():
    f = sn.snapshot_features(_rows(), sn.load_snapshots("does-not-exist.csv"))
    assert f[sn.FEATURES].isna().all().all() and (f[sn.COP] == 1).all()


@pytest.mark.parametrize("status", ["a", "d", "i", "s", "u", "n", "", None, "A", " d "])
@pytest.mark.parametrize("chance", [None, np.nan, 0, 25, 50, 75, 100, 150, -10])
def test_cop_factor_matches_availability_factor(status, chance):
    from pipeline.predict_next_gw import availability_factor
    assert sn.cop_factor([status], [chance])[0] == pytest.approx(availability_factor(status, chance))


def test_cop_post_hook_scales_every_protocols_predictions():
    from scripts.analysis import ml_backtest as bt
    frame = pd.DataFrame({sn.COP: [1.0, 0.5, 0.0]})
    assert bt._post(frame, [2.0, 4.0, 6.0], {"post": "cop"}).tolist() == [2.0, 2.0, 0.0]
    assert bt._post(frame, [2.0, 4.0, 6.0], {}).tolist() == [2.0, 4.0, 6.0]
    for name in ("cop", "avail_cop", "sp_cop", "snap_cop"):
        assert bt.VARIANTS[name]["post"] == "cop"
    assert set(bt.G_SNAP_AVAIL + bt.G_SP) <= set(bt.EXTRA)


def test_timing_check_counts_who_played_by_status():
    snaps = _snaps(_snap("2030-08-16T10:00:00Z", "s", {1: ("i", 0, 1, np.nan, np.nan, np.nan),
                                                        2: ("d", 75, 1, np.nan, np.nan, np.nan)}))
    rows = _rows()
    rows.loc[(rows["element"] == 1) & (rows["GW"] == 2), "minutes"] = 0.0
    out = sn.timing_check(rows, sn.snapshot_features(rows, snaps, with_status=True))
    assert out["out (i/s/u/n or chance 0)"] == (1, 0.0)       # one player-gameweek, both fixtures summed
    assert out["chance 75"] == (1, 1.0)
