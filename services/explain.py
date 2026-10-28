"""
SHAP and LIME explanations of the next-gameweek predictions, for the /model page
(GET /api/ml/explain/summary, /player, /dependence, /players).

Input: LIVE_DATA_DIR/next_gw_features.csv, the served model's input row for every upcoming
fixture, written by pipeline.predict_next_gw in the same run as next_gw_predictions.csv, and the
served model (ml.points_model.active_model_paths: v4, or v3 until a v4 file exists; the v3 shadow
isn't explained). Nothing is computed by the pipeline itself.

  SHAP  XGBoost's own TreeSHAP (Booster.predict(pred_contribs=True)): exact Shapley values of
        every feature for every row, in points. They add up, with the base value (the model's
        average output), to the model's raw output for that fixture. Computed for all rows at once
        and cached until the file or the model changes; the global views (mean |SHAP| per feature
        and per group, the beeswarm, dependence) come from that matrix.
  LIME  ml.lime_tabular: a weighted linear surrogate fitted to the model's outputs on 5000
        perturbed copies of one row, with features resampled from this gameweek's rows. Seeded per
        row, so the same player gets the same explanation until the inputs change.

The number the rest of the site shows is the predictions file's: the model's output floored at 0
per fixture, summed over a double gameweek, times FPL's chance of playing. The player view returns
those steps too.
"""
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml import lime_tabular
from ml import points_model as pm
from services.predictions import get_predictions_df
from utils.text import normalize_name

FEATURES_NAME = "next_gw_features.csv"
BEESWARM_FEATURES = 10
BEESWARM_MAX_ROWS = 500
TOP_PLAYERS = 12
LIME_SAMPLES = 5000
LIME_FEATURES = 10
BASELINES = ("last3_avg", "last6_avg", "season_ppg")   # shown after the served model's own metrics


class ExplainUnavailable(Exception):
    """No model inputs to explain (no prediction run yet) or they don't match the served model."""


def features_path() -> str:
    return os.path.join(os.getenv("LIVE_DATA_DIR", os.path.join("data", "live")), FEATURES_NAME)


# ------------------------------------------------------------------ feature descriptions

_STATS = {
    "total_points": ("Points", "FPL points"), "minutes": ("Minutes", "minutes played"),
    "started": ("Starts", "share of matches started"), "played": ("Appearances", "share of matches played"),
    "goals_scored": ("Goals", "goals"), "assists": ("Assists", "assists"), "bonus": ("Bonus", "bonus points"),
    "bps": ("BPS", "bonus points system score"), "ict_index": ("ICT index", "ICT index"),
    "influence": ("Influence", "influence"), "creativity": ("Creativity", "creativity"),
    "threat": ("Threat", "threat"), "clean_sheets": ("Clean sheets", "clean sheets"),
    "goals_conceded": ("Goals conceded", "goals conceded while on the pitch"), "saves": ("Saves", "saves"),
    "expected_goals": ("xG", "expected goals"), "expected_assists": ("xA", "expected assists"),
    "expected_goal_involvements": ("xGI", "expected goal involvements"),
    "expected_goals_conceded": ("xGC", "expected goals conceded"),
    "defensive_contribution": ("Defensive actions", "defensive contributions (clearances, blocks, "
                               "interceptions, tackles, recoveries)"),
}
_TEAM = {"gf": ("goals for", "goals scored"), "ga": ("goals against", "goals conceded"),
         "xgf": ("xG for", "expected goals created"), "xga": ("xG against", "expected goals conceded")}
_POSITIONS = {"GK": "goalkeeper", "DEF": "defender", "MID": "midfielder", "FWD": "forward"}

# v4's additions (ml.points_model.v4_extra_features). The wording stays neutral: bookmakers'
# prices are described as forecasts, never as betting.
_RATING = ("on a log scale where 0 is an average Premier League side, adjusted for the opponents it has "
           "faced and updated after every gameweek from xG and goals")
_NO_ODDS = " Missing when no prices were published before the deadline."
_NO_SNAPSHOT = " Missing when FPL had no snapshot before the deadline."
_V4 = {
    "rt_att": ("Team attack rating", "Team ratings", f"His team's attack rating, {_RATING}."),
    "rt_def": ("Team defence rating", "Team ratings", f"His team's defence rating (higher concedes less), {_RATING}."),
    "rt_opp_att": ("Opponent attack rating", "Team ratings", f"The opponent's attack rating, {_RATING}."),
    "rt_opp_def": ("Opponent defence rating", "Team ratings",
                   f"The opponent's defence rating (higher concedes less), {_RATING}."),
    "rt_xgf": ("Expected team xG (ratings)", "Team ratings",
               "xG his team is expected to create in this fixture, from both teams' ratings and home advantage."),
    "rt_xga": ("Expected xG against (ratings)", "Team ratings",
               "xG his team is expected to concede in this fixture, from both teams' ratings and home advantage."),
    "rt_cs": ("Clean-sheet chance (ratings)", "Team ratings",
              "Chance his team concedes no goals, from the ratings' expected xG against, times the share of his "
              "last 6 matches he started."),
    "rt_pxg": ("His expected goals (ratings)", "Team ratings",
               "His share of his team's xG over his last 6 matches times the team's expected xG in this fixture."),
    "rt_pxa": ("His expected assists (ratings)", "Team ratings",
               "His xA as a share of his team's xG over his last 6 matches, times the team's expected xG in this "
               "fixture."),
    "tr_bal_rank": ("Net transfers percentile", "Transfers",
                    "FPL managers' transfers in minus out before this gameweek's deadline, as a percentile of this "
                    "gameweek's players (1 = most bought on balance)."),
    "tr_in_rank": ("Transfers in percentile", "Transfers",
                   "FPL managers' transfers in before this gameweek's deadline, as a percentile of this gameweek's "
                   "players (1 = most bought)."),
    "tr_out_rank": ("Transfers out percentile", "Transfers",
                    "FPL managers' transfers out before this gameweek's deadline, as a percentile of this "
                    "gameweek's players (1 = most sold)."),
    "price_change_1": ("Price change since last match (£m)", "Price & ownership",
                       "His FPL price now minus his price at his previous match this season."),
    "price_change_3": ("Price change over 3 matches (£m)", "Price & ownership",
                       "His FPL price now minus his average price over his last 3 matches this season."),
    "odds_win": ("Bookmakers' implied win chance", "Bookmaker forecasts",
                 "His team's chance of winning this fixture implied by bookmakers' prices before the deadline "
                 "(0 to 1, margin removed)." + _NO_ODDS),
    "odds_draw": ("Bookmakers' implied draw chance", "Bookmaker forecasts",
                  "The chance of a draw implied by bookmakers' prices before the deadline (0 to 1, margin removed)."
                  + _NO_ODDS),
    "odds_gf": ("Bookmakers' expected goals for", "Bookmaker forecasts",
                "Goals his team is expected to score in this fixture, worked out from bookmakers' prices for the "
                "result and for over/under 2.5 goals." + _NO_ODDS),
    "odds_ga": ("Bookmakers' expected goals against", "Bookmaker forecasts",
                "Goals his team is expected to concede in this fixture, worked out from bookmakers' prices for the "
                "result and for over/under 2.5 goals." + _NO_ODDS),
    "odds_cs": ("Bookmakers' clean-sheet chance", "Bookmaker forecasts",
                "Chance his team concedes no goals, from the bookmakers' expected goals against (0 to 1)." + _NO_ODDS),
    "snap_unavailable": ("Flagged unavailable", "Availability",
                         "1 if FPL listed him as injured, suspended, unavailable or not in the squad before the "
                         "deadline." + _NO_SNAPSHOT),
    "snap_doubtful": ("Flagged doubtful", "Availability",
                      "1 if FPL listed him as doubtful before the deadline." + _NO_SNAPSHOT),
    "snap_chance": ("FPL chance of playing (%)", "Availability",
                    "FPL's chance of him playing this gameweek as listed before the deadline (100 with no flag)."
                    + _NO_SNAPSHOT),
    "snap_news": ("Has team news", "Availability",
                  "1 if FPL had team news on him before the deadline (an injury, a suspension, a transfer)."
                  + _NO_SNAPSHOT),
    "snap_age_days": ("Age of availability news (days)", "Availability",
                      "How long before the deadline FPL's availability information was captured, in days."
                      + _NO_SNAPSHOT),
}


def feature_info(feature: str) -> Tuple[str, str, str]:
    """(label, group, description) of a model feature."""
    for w in pm.WINDOWS:
        for stat, (label, words) in _STATS.items():
            if feature == f"{stat}_r{w}":
                return (f"{label}, last {w} matches", "Recent form",
                        f"Average {words} per match over the player's last {w} matches before this gameweek "
                        f"(matches he missed count as zero).")
    last = {"minutes_r1": ("Minutes in last match", "Minutes he played in his previous match."),
            "started_r1": ("Started last match", "Whether he started his previous match (1) or not (0)."),
            "total_points_r1": ("Points in last match", "His FPL points in his previous match.")}
    if feature in last:
        return last[feature][0], "Last match", last[feature][1]
    season = {
        "season_matches_before": ("Matches this season", "His team's matches this season before this gameweek."),
        "season_ppg": ("Points per match this season", "His FPL points per match so far this season."),
        "season_minutes_share": ("Share of minutes this season", "Minutes played as a share of all available "
                                 "minutes so far this season."),
        "career_matches_before": ("Career matches", "Premier League matches in the data since 2020-21."),
    }
    if feature in season:
        return season[feature][0], "Season & career", season[feature][1]
    for prefix, who, group in (("team", "Team", "Team strength"), ("opp", "Opponent", "Opponent strength")):
        for w in pm.TEAM_WINDOWS:
            for stat, (label, words) in _TEAM.items():
                if feature == pm._team_feature(prefix, stat, w):
                    whose = "his team" if prefix == "team" else "the opponent"
                    return (f"{who} {label}, last {w}", group,
                            f"Average {words} per match by {whose} over its last {w} matches.")
    other = {
        "days_rest": ("Days since last match", "Fixture", "Days since his previous match (capped at 30)."),
        "fixtures_in_gw": ("Fixtures this gameweek", "Fixture", "2 in a double gameweek."),
        "was_home": ("Home match", "Fixture", "1 at home, 0 away."),
        "price": ("Price (£m)", "Price & ownership", "His FPL price before the deadline."),
        "selected_rank": ("Ownership percentile", "Price & ownership",
                          "How widely owned he is, as a percentile of this gameweek's players (1 = most owned)."),
    }
    if feature in other:
        return other[feature]
    if feature.startswith("pos_"):
        pos = _POSITIONS.get(feature[4:], feature[4:])
        return f"Is a {pos}", "Position", f"1 if he is listed as a {pos}, else 0."
    if feature in _V4:
        return _V4[feature]
    return feature, "Other", ""


def _round(v: Any, digits: int = 4) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), digits)


# ------------------------------------------------------------------ cached state

@dataclass
class _State:
    key: tuple
    rows: pd.DataFrame          # id columns + model_points, one per upcoming fixture
    x: np.ndarray               # model inputs, rows x features
    features: List[str]
    shap: np.ndarray            # rows x features
    base: float
    raw: np.ndarray             # model output per row (base + shap sum)
    booster: Any
    meta: Dict[str, Any]
    model_source: str


_lock = threading.Lock()
_state: Optional[_State] = None
_lime_cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_lime_lock = threading.Lock()   # routes run in FastAPI's threadpool
_LIME_CACHE_SIZE = 256


def _key() -> tuple:
    path = features_path()
    model_path, meta_path = pm.active_model_paths()
    try:
        return path, os.path.getmtime(path), model_path, os.path.getmtime(model_path), os.path.getmtime(meta_path)
    except OSError:
        raise ExplainUnavailable("Explanations appear after the next prediction run.") from None


def _load() -> _State:
    global _state
    key = _key()
    with _lock:
        if _state is not None and _state.key == key:
            return _state
        import xgboost as xgb

        frame = pd.read_csv(key[0], float_precision="round_trip")   # exact, as written
        booster, meta = pm.load_model(key[2], meta_path=pm.active_model_paths()[1])
        features = list(meta["features"])
        missing = [c for c in features if c not in frame.columns]
        if missing or frame.empty:
            raise ExplainUnavailable("The saved model inputs don't match the current model yet; "
                                     "they're rewritten on the next prediction run.")
        x = frame[features].to_numpy(dtype=float)
        contribs = booster.predict(xgb.DMatrix(x, feature_names=features), pred_contribs=True)
        # Explain the numbers the site shows: if the model changed after the prediction run
        # (a retrain without a new run), its outputs no longer match the saved ones
        if "model_points" in frame and np.nanmax(np.abs(contribs.sum(axis=1) - frame["model_points"])) > 0.01:
            raise ExplainUnavailable("The model changed after the last prediction run; "
                                     "explanations return with the next run.")
        ids = [c for c in frame.columns if c not in set(features)]
        source = "live" if os.path.abspath(key[2]).startswith(os.path.abspath(pm.live_model_dir())) else "bundled"
        _state = _State(key=key, rows=frame[ids].reset_index(drop=True), x=x, features=features,
                        shap=contribs[:, :-1], base=float(contribs[0, -1]), raw=contribs.sum(axis=1),
                        booster=booster, meta=meta, model_source=source)
        with _lime_lock:
            _lime_cache.clear()
        return _state


def _predict_fn(state: _State):
    import xgboost as xgb
    return lambda m: state.booster.predict(xgb.DMatrix(m, feature_names=state.features))


# ------------------------------------------------------------------ views

def _model_info(state: _State) -> Dict[str, Any]:
    meta = state.meta
    metrics = meta.get("test_metrics") or {}
    version = str(meta.get("version", "v3"))
    return {
        "version": version, "trained_at": meta.get("trained_at"),
        "n_trees": meta.get("n_trees"), "seasons": list(meta.get("seasons") or []), "rows": meta.get("rows"),
        "source": state.model_source,
        # The model's own metrics are under its version ("v4"), as in its report
        "test_metrics": {k: metrics[k] for k in (version,) + BASELINES if isinstance(metrics.get(k), dict)},
    }


def _predictions_by_element() -> Dict[int, Dict[str, Any]]:
    df = get_predictions_df()
    if df is None or "element" not in df.columns:
        return {}
    out = {}
    for r in df.to_dict(orient="records"):
        try:
            out[int(r["element"])] = r
        except (TypeError, ValueError):
            continue
    return out


def _player_hits(state: _State, elements: List[int]) -> List[Dict[str, Any]]:
    preds = _predictions_by_element()
    first = state.rows.drop_duplicates("element").set_index("element")
    hits = []
    for e in elements:
        r = first.loc[e]
        pred = preds.get(int(e), {})
        points = pred.get("predicted_points")
        if points is None:
            points = float(np.clip(state.raw[(state.rows["element"] == e).to_numpy()], 0, None).sum())
        hits.append({"name": r["name"], "element": int(e), "team": r["team"], "position": r["position"],
                     "predicted_points": round(float(points), 2)})
    return hits


def _ranked_elements(state: _State) -> List[int]:
    """Every element in the inputs, by predicted points (the site's number), then name."""
    preds = _predictions_by_element()
    names = state.rows.drop_duplicates("element").set_index("element")["name"].astype(str)
    model = pd.Series(np.clip(state.raw, 0, None)).groupby(state.rows["element"].to_numpy()).sum()

    def points(e):
        p = preds.get(int(e), {}).get("predicted_points")
        return float(p) if p is not None and not pd.isna(p) else float(model.get(e, 0.0))

    return sorted((int(e) for e in names.index), key=lambda e: (-points(e), names.loc[e]))


def summary() -> Dict[str, Any]:
    state = _load()
    abs_mean = np.abs(state.shap).mean(axis=0)
    infos = {f: feature_info(f) for f in state.features}
    features = [{"feature": f, "label": infos[f][0], "group": infos[f][1], "description": infos[f][2],
                 "importance": round(float(abs_mean[j]), 4)} for j, f in enumerate(state.features)]
    features.sort(key=lambda d: -d["importance"])

    groups: Dict[str, np.ndarray] = {}
    for j, f in enumerate(state.features):
        g = infos[f][1]
        groups[g] = groups.get(g, 0.0) + state.shap[:, j]
    group_rows = sorted(({"group": g, "importance": round(float(np.abs(v).mean()), 4)} for g, v in groups.items()),
                        key=lambda d: -d["importance"])

    n = len(state.rows)
    sample = np.arange(n)
    if n > BEESWARM_MAX_ROWS:
        sample = np.sort(np.random.default_rng(0).choice(n, BEESWARM_MAX_ROWS, replace=False))
    index = {f: j for j, f in enumerate(state.features)}
    beeswarm = []
    for d in features[:BEESWARM_FEATURES]:
        j = index[d["feature"]]
        pct = pd.Series(state.x[:, j]).rank(pct=True).to_numpy()
        beeswarm.append({"feature": d["feature"], "label": d["label"],
                         "points": [[_round(pct[i], 3), round(float(state.shap[i, j]), 3)] for i in sample]})

    first = state.rows.iloc[0]
    return {
        "season": str(first["season"]), "gameweek": int(first["GW"]),
        "generated_at": pd.Timestamp(state.key[1], unit="s", tz="UTC").isoformat(),
        "base_value": round(state.base, 4), "rows": n, "players": int(state.rows["element"].nunique()),
        "model": _model_info(state), "groups": group_rows, "features": features, "beeswarm": beeswarm,
        "top_players": _player_hits(state, _ranked_elements(state)[:TOP_PLAYERS]),
    }


def search(q: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Players in this gameweek's inputs whose name contains `q` (accents and case ignored)."""
    state = _load()
    needle = normalize_name(q or "")
    ranked = _ranked_elements(state)
    if needle:
        names = state.rows.drop_duplicates("element").set_index("element")["name"]
        ranked = [e for e in ranked if needle in normalize_name(str(names.loc[e]))]
    return _player_hits(state, ranked[:limit])


def find_element(name: str) -> Optional[int]:
    """The element of the player called `name` (exact, accents and case ignored) in this gameweek's inputs."""
    state = _load()
    target = normalize_name(name)
    for e, n in state.rows.drop_duplicates("element")[["element", "name"]].itertuples(index=False):
        if normalize_name(str(n)) == target:
            return int(e)
    return None


def _lime(state: _State, i: int, labels: List[str]) -> Dict[str, Any]:
    key = (state.key, i)
    with _lime_lock:
        if key in _lime_cache:
            _lime_cache.move_to_end(key)
            return _lime_cache[key]
    row = state.rows.iloc[i]
    seed = int(row["element"]) * 1000 + int(row["fixture"]) % 1000
    res = lime_tabular.explain(state.x[i], state.x, _predict_fn(state), state.features, labels,
                               num_samples=LIME_SAMPLES, num_features=LIME_FEATURES, seed=seed)
    out = {
        "intercept": round(res.intercept, 4), "local_prediction": round(res.local_prediction, 4),
        "model_prediction": round(res.model_prediction, 4), "score": round(res.score, 4),
        "num_samples": res.num_samples, "kernel_width": round(res.kernel_width, 3),
        "weights": [{"feature": state.features[j], "label": labels[j], "condition": c,
                     "weight": round(w, 4), "value": _round(state.x[i, j])}
                    for j, w, c in zip(res.features, res.weights, res.conditions)],
    }
    with _lime_lock:
        _lime_cache[key] = out
        if len(_lime_cache) > _LIME_CACHE_SIZE:
            _lime_cache.popitem(last=False)
    return out


def player(element: int) -> Dict[str, Any]:
    """SHAP and LIME for each of the player's fixtures this gameweek; KeyError if he has none."""
    state = _load()
    idx = np.flatnonzero(state.rows["element"].to_numpy() == int(element))
    if not len(idx):
        raise KeyError(element)
    infos = [feature_info(f) for f in state.features]
    labels = [i[0] for i in infos]
    fixtures = []
    for i in idx:
        r = state.rows.iloc[i]
        order = np.argsort(-np.abs(state.shap[i]), kind="mergesort")
        fixtures.append({
            "fixture": int(r["fixture"]), "opponent": str(r["opponent"]),
            "is_home": bool(state.x[i, state.features.index("was_home")] > 0) if "was_home" in state.features else False,
            "kickoff_time": None if pd.isna(r["kickoff_time"]) else str(r["kickoff_time"]),
            "base_value": round(state.base, 4), "model_points": round(float(state.raw[i]), 4),
            "contributions": [{"feature": state.features[j], "label": infos[j][0], "group": infos[j][1],
                               "value": _round(state.x[i, j]), "shap": round(float(state.shap[i, j]), 4)}
                              for j in order],
            "lime": _lime(state, int(i), labels),
        })
    fixtures.sort(key=lambda f: (f["kickoff_time"] or "", f["fixture"]))
    first = state.rows.iloc[idx[0]]
    model_points = float(np.clip(state.raw[idx], 0, None).sum())
    pred = _predictions_by_element().get(int(element), {})
    from pipeline.predict_next_gw import availability_factor

    status = pred.get("status")
    chance = pred.get("chance_of_playing_next_round")
    factor = availability_factor(status, None if chance is None or pd.isna(chance) else chance)
    predicted = pred.get("predicted_points")
    return {
        "player": {"name": str(first["name"]), "element": int(element), "team": str(first["team"]),
                   "position": str(first["position"]),
                   "status": None if status is None or pd.isna(status) else str(status),
                   "chance_of_playing_next_round": _round(chance, 0) if chance is not None else None,
                   "news": pred.get("news") if isinstance(pred.get("news"), str) else ""},
        "season": str(first["season"]), "gameweek": int(first["GW"]), "fixtures": fixtures,
        "model_points": round(model_points, 4), "availability_factor": round(float(factor), 4),
        "predicted_points": round(float(predicted), 2) if predicted is not None and not pd.isna(predicted)
        else round(model_points * factor, 2),
    }


def dependence(feature: str) -> Dict[str, Any]:
    """Feature value against its SHAP value for every row; KeyError for an unknown feature."""
    state = _load()
    if feature not in state.features:
        raise KeyError(feature)
    j = state.features.index(feature)
    label, group, description = feature_info(feature)
    points = [{"name": str(r["name"]), "element": int(r["element"]), "position": str(r["position"]),
               "value": _round(state.x[i, j]), "shap": round(float(state.shap[i, j]), 4)}
              for i, r in enumerate(state.rows[["name", "element", "position"]].to_dict(orient="records"))]
    return {"feature": feature, "label": label, "group": group, "description": description, "points": points}


def reset() -> None:
    """Forget the cached inputs and explanations (tests; the cache also follows file mtimes)."""
    global _state
    with _lock, _lime_lock:
        _state = None
        _lime_cache.clear()
