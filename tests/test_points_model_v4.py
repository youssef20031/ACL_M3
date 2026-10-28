"""v4 features of ml/points_model.py (ExtraInputs): training/serving parity and leakage guarantees,
on synthetic data (no files apart from tmp_path)."""
import os

import numpy as np
import pandas as pd
import pytest

from ml import fixture_strength as fs
from ml import market as mk
from ml import odds as od
from ml import points_model as pm
from ml import snapshots as sn

TEAMS = ["Arsenal", "Chelsea", "Leeds", "Spurs"]
# GW1 Saturdays; one match per team per gameweek, a week apart
SEASON_START = {"2024-25": "2024-08-17", "2025-26": "2025-08-16", "2026-27": "2026-08-22"}
PAIRS = {1: [(0, 1), (2, 3)], 2: [(0, 2), (1, 3)], 0: [(0, 3), (1, 2)]}
# (name in 2024-25 and 2025-26, name in 2026-27, team, position, code); elements change by season
PLAYERS = [("Alice", "Alice", "Arsenal", "MID", 100), ("Bob", "Bob", "Chelsea", "FWD", 200),
           ("Lukasz F", "Łukasz F", "Chelsea", "DEF", 300), ("Lee", "Lee", "Leeds", "DEF", 400),
           ("Sam", "Sam", "Spurs", "GK", 500)]
GWS = 4   # per season; 2026-27 GW4 is "the upcoming gameweek" in the parity tests


def _element(season, code):
    return code // 100 + 10 * list(SEASON_START).index(season)


def _kickoff(season, gw):
    return pd.Timestamp(SEASON_START[season] + "T14:00:00Z") + pd.Timedelta(days=7 * (gw - 1))


def _history():
    """Every match of three seasons, GWS gameweeks each, with elements, transfers and xG."""
    rows, fixture = [], 0
    for season in SEASON_START:
        for gw in range(1, GWS + 1):
            for i, (h, a) in enumerate(PAIRS[gw % 3]):
                fixture += 1
                home, away = (TEAMS[h], TEAMS[a]) if gw % 2 else (TEAMS[a], TEAMS[h])
                hg, ag = (gw * 7 + i * 3 + fixture) % 4, (gw * 5 + i) % 3
                for old, new, team, pos, code in PLAYERS:
                    if team not in (home, away):
                        continue
                    el = _element(season, code)
                    t_in, t_out = 100 * ((gw * el) % 7), 50 * ((gw + el) % 5)
                    minutes = 0 if (gw + el) % 6 == 0 else 90
                    rows.append({
                        "season": season, "name": new if season == "2026-27" else old, "position": pos,
                        "team": team, "fixture": fixture, "kickoff_time": _kickoff(season, gw), "GW": gw,
                        "was_home": team == home, "team_h_score": hg, "team_a_score": ag,
                        "total_points": (gw + el) % 9 if minutes else 0, "minutes": minutes,
                        "starts": float(minutes > 0), "goals_scored": 0, "assists": 0, "bonus": 0, "bps": 10,
                        "ict_index": 5.0, "influence": 10.0, "creativity": 10.0, "threat": 10.0,
                        "clean_sheets": 0, "goals_conceded": 0, "saves": 0,
                        "expected_goals": 0.1 * ((gw + el) % 5), "expected_assists": 0.05 * (gw % 3),
                        "value": 50 + gw + el % 3, "selected": 1000 * el + gw, "element": el,
                        "transfers_in": t_in, "transfers_out": t_out, "transfers_balance": t_in - t_out,
                    })
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[pm.HISTORY_COLUMNS].copy()


def _codes():
    return pd.DataFrame([{"season": s, "element": _element(s, p[4]), "code": p[4]}
                         for s in SEASON_START for p in PLAYERS])


def _odds(history):
    """football-data.co.uk rows for every match of `history`, then parsed (prices vary by match)."""
    m = history.drop_duplicates(["season", "fixture"])
    raw = []
    for r in m.itertuples(index=False):
        teams = history.loc[history["fixture"] == r.fixture, ["team", "was_home"]].drop_duplicates()
        home = teams.loc[teams["was_home"], "team"].iloc[0]
        away = teams.loc[~teams["was_home"], "team"].iloc[0]
        k = r.fixture % 5
        raw.append({"Div": "E0", "Date": r.kickoff_time.strftime("%d/%m/%Y"), "HomeTeam": home, "AwayTeam": away,
                    "AvgH": 1.6 + 0.4 * k, "AvgD": 3.5, "AvgA": 5.0 - 0.5 * k, "Avg>2.5": 1.8 + 0.1 * k,
                    "Avg<2.5": 2.0 - 0.05 * k, "season": r.season})
    raw = pd.DataFrame(raw)
    return pd.concat([od.parse_odds(g, s) for s, g in raw.groupby("season")], ignore_index=True)


def _snapshot_rows(season, at, chance_by_code):
    """Bootstrap-like rows of every player at `at`: chance None = available."""
    rows = []
    for _, _, _, _, code in PLAYERS:
        chance = chance_by_code.get(code)
        status = "a" if chance is None else ("i" if chance == 0 else "d")
        rows.append({"element": _element(season, code), "code": code, "status": status,
                     "chance_of_playing_next_round": chance, "news": "" if chance is None else "Knock",
                     "penalties_order": 1 if code == 100 else np.nan})
    return pd.DataFrame(rows).assign(captured_at=at, fetched_at=at)


def _vaastav(history):
    """A snapshot two days before every gameweek's first kickoff (FPL's history), with a doubt or two."""
    frames = []
    for (season, gw), k in history.groupby(["season", "GW"])["kickoff_time"].min().items():
        rows = _snapshot_rows(season, k - pd.Timedelta(days=2), {200: 75} if gw % 2 else {400: 0})
        frames.append(sn.from_bootstrap(rows, season, "captured_at", f"v{season}-{gw}"))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def world():
    full = _history()
    gw4 = ((full["season"] == "2026-27") & (full["GW"] == 4)).to_numpy()
    # The status file, fetched the day before GW4's deadline (Saturday 12:30)
    status = _snapshot_rows("2026-27", _kickoff("2026-27", 4) - pd.Timedelta(hours=26), {100: 50, 300: 0})
    status["fetched_at"] = status["fetched_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"full": full, "gw4": gw4, "codes": _codes(), "odds": _odds(full), "vaastav": _vaastav(full[~gw4]),
            "status": status}


def _upcoming(rows: pd.DataFrame) -> pd.DataFrame:
    """GW4 rows as serving builds them: no outcomes, FPL's transfers in/out since the last deadline,
    the current price, ownership as a rank within the gameweek."""
    opp = pm._opponents(rows)
    up = rows.merge(opp, on=["season", "fixture", "team"])
    up["selected_rank"] = up.groupby(["season", "GW"])["selected"].rank(pct=True)
    return up[["season", "name", "position", "team", "opponent", "fixture", "kickoff_time", "GW", "was_home",
               "value", "selected_rank", "element", "transfers_in", "transfers_out"]]


def _key(f):
    return f.set_index(["name", "fixture"])


# --------------------------------------------------------------------------- feature lists

def test_v4_columns_are_v3_plus_the_new_groups():
    v3, v4 = pm.feature_columns("v3"), pm.feature_columns("v4")
    assert v3 == pm.V3_FEATURES and len(v3) == 72 and v3[0] == "total_points_r3" and v3[-1] == "opp_xga20"
    assert pm.feature_columns() == v4 and v4[:len(v3)] == v3 and len(set(v4)) == len(v4)
    assert v4[len(v3):] == fs.RATING_FEATURES + mk.FEATURES + od.FEATURES + sn.AVAIL_FEATURES
    assert not set(sn.SP_FEATURES) & set(v4)
    with pytest.raises(ValueError):
        pm.feature_columns("v5")


def test_without_extras_the_features_are_v3s(world):
    h = world["full"]
    plain = pm.build_features(h)
    assert (plain["player_key"] == plain["name"]).all()
    assert not set(pm.v4_extra_features()) & set(plain.columns)
    # v3 callers whose history has no element or transfer columns get the same values
    old = pm.build_features(h.drop(columns=["element", "transfers_in", "transfers_out", "transfers_balance"]))
    pd.testing.assert_frame_equal(plain[pm.V3_FEATURES], old[pm.V3_FEATURES])
    # Extras whose codes rename nobody change no v3 value
    same = h[h["name"] != "Łukasz F"]
    with_codes = pm.build_features(same, extras=pm.ExtraInputs(codes=world["codes"]))
    pd.testing.assert_frame_equal(pm.build_features(same)[pm.V3_FEATURES], with_codes[pm.V3_FEATURES])


# --------------------------------------------------------------------------- identity

def test_code_identity_joins_a_renamed_history(world):
    h = world["full"]
    by_name = _key(pm.build_features(h))
    by_code = _key(pm.build_features(h, extras=pm.ExtraInputs(codes=world["codes"])))
    first = h[(h["name"] == "Łukasz F")]["fixture"].min()
    earlier = h[h["name"] == "Lukasz F"].sort_values("kickoff_time")
    assert by_name.loc[("Łukasz F", first), "career_matches_before"] == 0   # the name restarts his history
    assert by_code.loc[("Łukasz F", first), "career_matches_before"] == len(earlier)
    assert by_code.loc[("Łukasz F", first), "total_points_r3"] == pytest.approx(earlier["total_points"].tail(3).mean())
    assert by_code.loc[("Łukasz F", first), "total_points_r1"] == earlier["total_points"].iloc[-1]
    # The display name stays; the key is one per player
    assert {"Lukasz F", "Łukasz F"} <= set(by_code.index.get_level_values("name"))
    assert by_code.loc[("Lukasz F", earlier["fixture"].iloc[0]), "player_key"] == "Łukasz F"


def test_player_identity_follows_the_code():
    rows = pd.DataFrame({"season": ["2024-25", "2025-26", "2025-26", "2025-26", "2025-26"],
                         "element": [7, 9, 1, 2, 3],
                         "name": ["Lukasz F", "Łukasz F", "Ben Davies", "Ben Davies", "No Code"]})
    codes = pd.DataFrame({"season": ["2024-25", "2025-26", "2025-26", "2025-26"], "element": [7, 9, 1, 2],
                          "code": [100, 100, 200, 300]})
    assert pm.player_identity(rows, codes).tolist() == ["Łukasz F", "Łukasz F", "Ben Davies #200",
                                                        "Ben Davies #300", "No Code"]


def test_player_codes_add_the_status_files_season(tmp_path):
    (tmp_path / "codes.csv").write_text("season,element,code\n2025-26,1,10\n", encoding="utf-8")
    (tmp_path / "status.csv").write_text("element,code,name\n5,50,X\n", encoding="utf-8")
    codes = pm.load_player_codes(str(tmp_path / "status.csv"), "2026-27", codes_csv=str(tmp_path / "codes.csv"))
    assert codes.values.tolist() == [["2025-26", 1, 10], ["2026-27", 5, 50]]


# --------------------------------------------------------------------------- serving parity

def test_upcoming_rows_get_what_the_same_rows_get_as_history(world):
    """GW4 predicted before its deadline (serving) = GW4 built as history from information known
    at the deadline (training): every v4 feature, and the chance-of-playing factor."""
    full, gw4 = world["full"], world["gw4"]
    serving = pm.ExtraInputs(codes=world["codes"], odds=world["odds"],
                             snapshots=sn.load_all_snapshots(os.devnull, None, world["status"], "2026-27"))
    up = pm.build_features(full[~gw4], _upcoming(full[gw4]), extras=serving).query("is_upcoming")
    # In training the same status capture is in our own archive
    archived = sn.from_bootstrap(world["status"], "2026-27", "fetched_at", "live")
    training = pm.ExtraInputs(codes=world["codes"], odds=world["odds"],
                              snapshots=pd.concat([world["vaastav"], archived], ignore_index=True))
    hist = pm.build_features(full, extras=training)
    hist = hist[(hist["season"] == "2026-27") & (hist["GW"] == 4)]
    cols = pm.feature_columns("v4") + [sn.COP]
    a, b = _key(up)[cols].sort_index(), _key(hist)[cols].sort_index()
    assert len(a) == len(PLAYERS) and a.index.equals(b.index)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    # ... and they carry real values: the status file's doubt, net transfers, odds, ratings
    assert a.loc[("Alice", a.loc["Alice"].index[0]), "snap_chance"] == 50 and a["cop_factor"].min() == 0
    assert a[["tr_bal_rank", "price_change_1", "odds_win", "rt_xgf", "snap_age_days"]].notna().all().all()
    assert a["snap_age_days"].iloc[0] == pytest.approx((26 * 60 - 90) / 1440)


def test_a_new_seasons_first_gameweek_with_a_promoted_club(world):
    """GW1 predicted before the season (no match of it in the history yet) = GW1 as history: the
    promoted club starts at RATING_NEW_TEAM in both, not regressed a second time over the summer."""
    full = world["full"].copy()
    new = full["season"] == "2026-27"
    full.loc[new & (full["team"] == "Leeds"), "team"] = "Hull City"
    gw1 = (new & (full["GW"] == 1)).to_numpy()
    extras = pm.ExtraInputs(codes=world["codes"], odds=_odds(full))
    up = pm.build_features(full[~new.to_numpy()], _upcoming(full[gw1]), extras=extras).query("is_upcoming")
    hist = pm.build_features(full, extras=extras)
    hist = hist[(hist["season"] == "2026-27") & (hist["GW"] == 1)]
    cols = pm.feature_columns("v4")
    a, b = _key(up)[cols].sort_index(), _key(hist)[cols].sort_index()
    assert len(a) == len(PLAYERS) and a.index.equals(b.index)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    hull = up[up["team"] == "Hull City"].iloc[0]
    assert hull["rt_att"] == hull["rt_def"] == fs.RATING_NEW_TEAM
    assert up.loc[up["opponent"] == "Hull City", "rt_opp_att"].iloc[0] == fs.RATING_NEW_TEAM


def test_upcoming_fixtures_without_a_kickoff_time(world):
    """A fixture in the gameweek whose kickoff isn't set yet gets no odds instead of stopping the
    prediction run (v3 never needed the time), and so does a gameweek without any kickoff time."""
    full, gw4 = world["full"], world["gw4"]
    extras = pm.ExtraInputs(codes=world["codes"], odds=world["odds"],
                            snapshots=sn.load_all_snapshots(os.devnull, None, world["status"], "2026-27"))
    up = _upcoming(full[gw4])
    tbc = (up["fixture"] == up["fixture"].max()).to_numpy()
    up.loc[tbc, "kickoff_time"] = None
    f = pm.build_features(full[~gw4], up, extras=extras).query("is_upcoming")
    late = (f["fixture"] == up["fixture"].max()).to_numpy()
    assert f.loc[late, od.FEATURES].isna().all().all() and f.loc[~late, od.FEATURES].notna().all().all()
    assert f["snap_age_days"].notna().all()            # the deadline comes from the known kickoffs
    up["kickoff_time"] = None
    f = pm.build_features(full[~gw4], up, extras=extras).query("is_upcoming")
    assert f[od.FEATURES + sn.AVAIL_FEATURES].isna().all().all() and (f[sn.COP] == 1).all()
    assert f[["tr_bal_rank", "rt_att"]].notna().all().all()
    assert od.collected_at(pd.Series([pd.NaT, pd.Timestamp("2026-08-22T14:00:00Z")])).isna().tolist() == [True, False]


def test_upcoming_transfer_ranks_use_in_minus_out(world):
    full, gw4 = world["full"], world["gw4"]
    up = _upcoming(full[gw4])
    f = pm.build_features(full[~gw4], up, extras=pm.ExtraInputs(codes=world["codes"])).query("is_upcoming")
    bal = (f["transfers_in"] - f["transfers_out"]).to_numpy()
    assert np.array_equal(f["transfers_balance"].to_numpy(), bal)
    assert np.array_equal(f["tr_bal_rank"].to_numpy(), pd.Series(bal).rank(pct=True).to_numpy())


# --------------------------------------------------------------------------- leakage

def test_odds_load_season_files_and_upcoming_fixtures(tmp_path):
    """fixtures.csv (all leagues, BOM) adds the upcoming Premier League matches; a match also in a
    season file keeps the season file's prices."""
    cols = "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
    played = "E0,22/08/2026,15:00,Arsenal,Chelsea,2.0,3.5,4.0,1.9,1.9,1.8,3.6,4.5,1.9,1.9\n"
    (tmp_path / "E0_2627.csv").write_text(cols + played, encoding="utf-8")
    (tmp_path / "fixtures.csv").write_text(
        "﻿" + cols
        + "E0,22/08/2026,15:00,Arsenal,Chelsea,3.0,3.0,3.0,1.9,1.9,,,,,\n"      # stale copy of a played match
        + "E0,29/08/2026,15:00,Man United,Hull,2.0,3.5,4.0,2.0,1.8,,,,,\n"
        + "E2,29/08/2026,15:00,Wigan,Luton,2.0,3.5,4.0,2.0,1.8,,,,,\n"
        + "E0,02/09/2026,19:45,Coventry,Spurs,,,,,,,,,,\n", encoding="utf-8")   # no prices yet
    odds = od.load_odds(str(tmp_path)).drop_duplicates(["season", "home", "away"], keep="last")
    o = odds.set_index(["season", "home", "away"])
    assert len(o) == 3 and ("2026-27", "Wigan", "Luton") not in o.index
    assert o.loc[("2026-27", "Arsenal", "Chelsea"), "p_home"] == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    assert np.isfinite(o.loc[("2026-27", "Arsenal", "Chelsea"), "p_home_c"])
    assert np.isfinite(o.loc[("2026-27", "Man Utd", "Hull City"), "p_home"])
    assert np.isnan(o.loc[("2026-27", "Coventry City", "Spurs"), "p_home"])
    assert od.season_of(["30/06/2027", "01/07/2027"]).tolist() == ["2026-27", "2027-28"]
    assert od.load_odds(str(tmp_path / "missing")).empty


def test_upcoming_odds_collected_after_the_deadline_are_not_used(world):
    """A rearranged Tuesday match inside a Saturday gameweek: fixtures.csv may list its prices, but
    they're collected that Tuesday, after the deadline, so the row gets NaN as in training."""
    full, gw4 = world["full"], world["gw4"]
    up = _upcoming(full[gw4])
    late = up[up["fixture"] == up["fixture"].max()].index
    up.loc[late, "kickoff_time"] = up.loc[late, "kickoff_time"] + pd.Timedelta(days=3, hours=5)
    odds = world["odds"].copy()
    f = pm.build_features(full[~gw4], up, extras=pm.ExtraInputs(codes=world["codes"], odds=odds)).query("is_upcoming")
    tuesday = f["fixture"] == up["fixture"].max()
    assert f.loc[tuesday, od.FEATURES].isna().all().all()
    assert f.loc[~tuesday, od.FEATURES].notna().all().all()


def test_a_snapshot_after_the_deadline_is_never_used(world):
    full = world["full"]
    gw3 = (full["season"] == "2026-27") & (full["GW"] == 3)
    base = pm.ExtraInputs(codes=world["codes"], snapshots=world["vaastav"])
    before = pm.build_features(full, extras=base)
    # A capture 10 minutes after GW3's deadline (everyone ruled out), and today's status file
    late = _snapshot_rows("2026-27", _kickoff("2026-27", 3) - pd.Timedelta(minutes=80), {p[4]: 0 for p in PLAYERS})
    now = _snapshot_rows("2026-27", pd.Timestamp("2026-10-28T09:00:00Z"), {p[4]: 0 for p in PLAYERS})
    snaps = pd.concat([world["vaastav"], sn.from_bootstrap(late, "2026-27", "captured_at", "live"),
                       sn.from_bootstrap(now, "2026-27", "captured_at", "status")], ignore_index=True)
    after = pm.build_features(full, extras=pm.ExtraInputs(codes=world["codes"], snapshots=snaps))
    cols = sn.AVAIL_FEATURES + [sn.COP]
    m3 = ((before["season"] == "2026-27") & (before["GW"] == 3)).to_numpy()
    pd.testing.assert_frame_equal(before.loc[m3, cols], after.loc[m3, cols])
    assert gw3.sum() == m3.sum() and before.loc[m3, "snap_unavailable"].notna().all()
    # GW4 (deadline after the late capture) uses it: everyone out
    m4 = ((after["season"] == "2026-27") & (after["GW"] == 4)).to_numpy()
    assert (after.loc[m4, "snap_unavailable"] == 1).all() and (after.loc[m4, sn.COP] == 0).all()
    # The status file, captured after every finished gameweek's deadline, changes no history row
    pd.testing.assert_frame_equal(before.loc[~m4, cols], after.loc[~m4, cols])


def test_live_archive_snapshots_in_the_history_schema(tmp_path):
    d = tmp_path / "element_snapshots" / "2026-27"
    d.mkdir(parents=True)
    pd.DataFrame({"element": [1, 2, 1], "code": [10, 20, 10], "GW": 9,
                  "captured_at": ["2026-10-23T05:00:00+00:00"] * 2 + ["2026-10-23T17:00:00+00:00"],
                  "status": ["a", "d", "i"], "chance_of_playing_next_round": [np.nan, 75, 0],
                  "news": [np.nan, "Knock", "Hamstring"], "penalties_order": [1, np.nan, 1],
                  "corners_and_indirect_freekicks_order": np.nan, "direct_freekicks_order": 2,
                  "now_cost": 55}).to_csv(d / "gw09.csv", index=False)
    snaps = sn.load_live_snapshots(str(tmp_path))
    assert list(snaps.columns) == sn.COLUMNS and len(snaps) == 3
    assert snaps["sha"].nunique() == 2 and set(snaps["season"]) == {"2026-27"}
    assert snaps["news_flag"].tolist() == [0, 1, 1]
    assert str(snaps["captured_at"].dtype) == "datetime64[ns, UTC]"
    assert sn.load_live_snapshots(str(tmp_path / "none")).empty
    every = sn.load_all_snapshots(str(tmp_path / "no.csv"), str(tmp_path))
    assert len(every) == 3


# --------------------------------------------------------------------------- ratings parity

def test_ratings_from_the_last_seasons_with_a_prior_equal_the_full_historys(world, tmp_path):
    full, gw4 = world["full"], world["gw4"]
    up = _upcoming(full[gw4])
    hist = full[~gw4]
    whole = pm.build_features(hist, up, extras=pm.ExtraInputs())
    recent = hist[hist["season"] >= "2025-26"]
    # Serving: the prior from the history CSV (format of data/cleaned_merged_seasons_cleaned.csv)
    csv = tmp_path / "history.csv"
    hist[hist["season"] < "2026-27"].rename(columns={"season": "season_x", "team": "team_x"}).to_csv(csv, index=False)
    prior = fs.team_match_history(history_csv=str(csv))
    pd.testing.assert_frame_equal(prior, fs.team_match_history(hist[hist["season"] < "2026-27"]), check_dtype=False)
    served = pm.build_features(recent, up, extras=pm.ExtraInputs(ratings_prior=prior))
    without = pm.build_features(recent, up, extras=pm.ExtraInputs())
    on = ["name", "fixture", "season", "is_upcoming"]
    a = whole[whole["season"] >= "2025-26"].set_index(on)[fs.RATING_FEATURES].sort_index()
    b = served.set_index(on)[fs.RATING_FEATURES].sort_index()
    # The team ratings on every row; the player-share products (rt_cs, rt_pxg, rt_pxa) also use
    # 6-match player form, which a shorter history only changes near its start, not for GW4
    teams = fs.RATING_FEATURES[:6]
    pd.testing.assert_frame_equal(a[teams], b[teams])
    pd.testing.assert_frame_equal(a.xs(True, level="is_upcoming"), b.xs(True, level="is_upcoming"))
    c = without.set_index(on)[fs.RATING_FEATURES].sort_index()
    assert not np.allclose(a["rt_att"], c["rt_att"])     # without the prior, 2025-26 starts from scratch


# --------------------------------------------------------------------------- models

def test_model_paths_by_version(tmp_path, monkeypatch):
    bundled, live = tmp_path / "bundled", tmp_path / "live"
    bundled.mkdir()
    monkeypatch.setattr(pm, "MODEL_DIR", str(bundled))
    monkeypatch.setenv("LIVE_DATA_DIR", str(live))

    def touch(*paths):
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()

    touch(*pm.bundled_model_paths("v3"))
    assert pm.active_model_paths() == pm.bundled_model_paths("v3")     # no v4 yet: the shadow serves
    assert pm.active_model_paths("v4") == pm.bundled_model_paths("v4")
    touch(*pm.bundled_model_paths("v4"))
    assert pm.active_model_paths() == pm.bundled_model_paths("v4")
    touch(*pm.live_model_paths("v3"))
    assert pm.active_model_paths("v3") == pm.live_model_paths("v3")
    assert pm.active_model_paths() == pm.bundled_model_paths("v4")
    touch(*pm.live_model_paths("v4"))
    assert pm.active_model_paths() == pm.live_model_paths("v4") == (str(live / "models" / "points_v4.json"),
                                                                    str(live / "models" / "points_v4_meta.json"))


def test_save_and_load_by_version(world, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path / "live"))
    f = pm.build_features(world["full"], extras=pm.ExtraInputs(codes=world["codes"], odds=world["odds"],
                                                                snapshots=world["vaastav"]))
    for version in ("v3", "v4"):
        model = pm.fit(f, None, n_estimators=5, cols=pm.feature_columns(version))
        pm.save_model(model, {"version": version}, *pm.bundled_model_paths(version))
    booster, meta = pm.load_model(version="v3")
    assert meta["features"] == pm.V3_FEATURES
    v4_booster, v4_meta = pm.load_model()
    assert v4_meta["version"] == "v4" and v4_meta["features"] == pm.feature_columns("v4")
    # Both predict from the same v4 feature table
    assert len(pm.predict(booster, meta, f)) == len(pm.predict(v4_booster, v4_meta, f)) == len(f)


def test_training_refuses_a_finished_season_without_odds():
    """A failed download of one past season's file leaves the overall share above the bar; the
    season itself is caught. The season in progress may lag."""
    from ml import train_points_v3 as tp
    cov = {"2020-21": {"odds": 0.9}, "2021-22": {"odds": 0.0}, "2025-26": {"odds": 0.98},
           "2026-27": {"odds": 0.1}, "all": {"odds": 0.72}}
    assert tp.odds_gaps(cov) == {"2021-22": 0.0}
    cov["2021-22"]["odds"] = 0.88
    assert tp.odds_gaps(cov) == {}
    assert tp.odds_gaps({"2020-21": {"odds": 0.3}, "2021-22": {"odds": 0.4}, "all": {"odds": 0.35}}) == {
        "2020-21": 0.3, "all": 0.35}


def test_backtest_keeps_v3_as_current_and_builds_v4_with_pm():
    from scripts.analysis import ml_backtest as bt
    assert bt.CURRENT == pm.V3_FEATURES
    spec = bt.VARIANTS["v4"]
    assert spec["features"] == [f"{c}__v4" for c in pm.feature_columns("v4")] and spec["post"] == "cop"
    columns = bt.KEEP + bt.CURRENT + bt.EXTRA + bt.SUBSETS + bt.CODE_COLS + bt.V4
    assert not set(bt.V4) & set(bt.KEEP + bt.CURRENT + bt.EXTRA + bt.SUBSETS + bt.CODE_COLS)
    assert len(bt.V4) == len(set(bt.V4)) and set(bt.V4) <= set(columns)
    assert bt.player_identity is pm.player_identity
