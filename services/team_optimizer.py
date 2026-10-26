"""
Team-of-the-week selection: pure functions, no Neo4j or file IO.

  best_xi(pool)       best valid starting XI by points (no budget, no club limit), like
                      FPL's Dream Team. Exact greedy: take each position's minimum, then
                      fill the remaining slots with the best players left under the caps.
  best_squad(pool)    full 15-man FPL squad (2 GK, 5 DEF, 5 MID, 3 FWD) within a budget and
                      a per-club limit. Exact, via scipy's MILP solver (HiGHS). It maximises
                      the expected squad score: the XI's points, the captain's points again,
                      and each bench player's points weighted by how often that bench slot
                      comes on (BENCH_WEIGHTS). So spare cash buys a bench that plays, and
                      a little XI value is traded for a much better first substitute.
  apply_auto_subs()   FPL's automatic substitutions, for scoring a squad after the gameweek
                      (services/model_review, scripts/analysis/ml_backtest).

Costs are integers in FPL tenths (156 = 15.6m). Ties are deterministic: the pool is
sorted before solving and every ordering ends on the candidate id.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

POSITIONS: Tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
SQUAD_SIZE: Dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN: Dict[str, int] = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX: Dict[str, int] = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
DEFAULT_BUDGET = 1000      # tenths: 100.0m
DEFAULT_MAX_PER_CLUB = 3

# Share of a bench player's predicted points expected to count, by bench slot. A player
# who played 60+ minutes in each of his last two matches still gets no minutes in about
# 7% of the next ones (2024-25 to 2026-27 FPL data), so with ten outfield starters the
# first outfield substitute comes on about half the time, the second about one week in
# seven and the third rarely; a goalkeeper is almost never replaced.
BENCH_WEIGHTS: Tuple[float, float, float] = (0.5, 0.15, 0.05)   # outfield subs 1, 2, 3
BENCH_GK_WEIGHT = 0.05


class InfeasibleError(ValueError):
    """The pool can't fill a valid team under the constraints."""


@dataclass(frozen=True)
class Candidate:
    id: str            # unique key (player name)
    position: str      # GK / DEF / MID / FWD
    team: str
    points: float
    cost: int = 0      # FPL tenths


@dataclass
class Selection:
    starters: List[Candidate]          # GK, DEF, MID, FWD; points desc within a position
    bench: List[Candidate]             # GK first, then outfield by points desc
    captain: Candidate
    vice_captain: Optional[Candidate]

    @property
    def formation(self) -> str:
        return formation(self.starters)


# ------------------------------------------------------------------ ordering helpers

def _rank_key(c: Candidate) -> Tuple[float, str]:
    return (-c.points, c.id)


def order_starters(players: Iterable[Candidate]) -> List[Candidate]:
    return sorted(players, key=lambda c: (POSITIONS.index(c.position), -c.points, c.id))


def order_bench(players: Iterable[Candidate]) -> List[Candidate]:
    """FPL bench order: the goalkeeper first, then outfield players by points."""
    return sorted(players, key=lambda c: (c.position != "GK", -c.points, c.id))


def formation(starters: Sequence[Candidate]) -> str:
    counts = {p: 0 for p in POSITIONS}
    for c in starters:
        counts[c.position] += 1
    return "-".join(str(counts[p]) for p in ("DEF", "MID", "FWD"))


def pick_captains(starters: Sequence[Candidate]) -> Tuple[Candidate, Optional[Candidate]]:
    ranked = sorted(starters, key=_rank_key)
    if not ranked:
        raise InfeasibleError("No starters to captain")
    return ranked[0], (ranked[1] if len(ranked) > 1 else None)


def _check_positions(pool: Sequence[Candidate]) -> None:
    ids = [c.id for c in pool]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate ids must be unique")
    bad = {c.position for c in pool} - set(POSITIONS)
    if bad:
        raise ValueError(f"Unknown position(s): {sorted(bad)}")


# ------------------------------------------------------------------ best XI (no budget)

def best_xi(pool: Iterable[Candidate]) -> Selection:
    """Highest-scoring valid XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD); no bench.

    Greedy is exact here: the position minimums must be filled, and the remaining
    slots are interchangeable apart from the per-position caps, so taking the best
    remaining players under those caps is optimal.
    """
    pool = list(pool)
    _check_positions(pool)
    by_pos = {p: sorted((c for c in pool if c.position == p), key=_rank_key) for p in POSITIONS}
    for p in POSITIONS:
        if len(by_pos[p]) < XI_MIN[p]:
            raise InfeasibleError(f"Need at least {XI_MIN[p]} {p}, have {len(by_pos[p])}")

    chosen: List[Candidate] = []
    extras: List[Candidate] = []
    for p in POSITIONS:
        chosen.extend(by_pos[p][:XI_MIN[p]])
        extras.extend(by_pos[p][XI_MIN[p]:XI_MAX[p]])
    free = XI_SIZE - len(chosen)
    extras.sort(key=lambda c: (-c.points, POSITIONS.index(c.position), c.id))
    if len(extras) < free:
        raise InfeasibleError(f"Need {XI_SIZE} players, have {len(chosen) + len(extras)}")
    chosen.extend(extras[:free])

    starters = order_starters(chosen)
    captain, vice = pick_captains(starters)
    return Selection(starters=starters, bench=[], captain=captain, vice_captain=vice)


# ------------------------------------------------------------------ best squad (MILP)

def best_squad(pool: Iterable[Candidate], budget: int = DEFAULT_BUDGET,
               max_per_club: int = DEFAULT_MAX_PER_CLUB,
               time_limit: float = 20.0) -> Selection:
    """Best 15-man squad within budget/club limits, by expected points: XI + captain
    + bench weighted by BENCH_WEIGHTS / BENCH_GK_WEIGHT.

    Raises InfeasibleError when no valid squad exists.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    pool = sorted(pool, key=lambda c: (POSITIONS.index(c.position) if c.position in POSITIONS else 9,
                                       -c.points, c.cost, c.id))
    _check_positions(pool)
    for p in POSITIONS:
        have = sum(c.position == p for c in pool)
        if have < SQUAD_SIZE[p]:
            raise InfeasibleError(f"Need {SQUAD_SIZE[p]} {p}, have {have}")

    n = len(pool)
    points = np.array([c.points for c in pool], dtype=float)
    cost = np.array([c.cost for c in pool], dtype=float)
    clubs = sorted({c.team for c in pool})
    is_gk = np.array([c.position == "GK" for c in pool])

    # Variables, n each: x (in squad), s (starts), k (captain), f (first outfield sub),
    # g (second outfield sub). The other bench players are covered by x - s.
    X, S, K, F, G = (j * n for j in range(5))
    rows: List[Tuple[Dict[int, float], float, float]] = []
    for i in range(n):
        rows.append(({S + i: 1.0, X + i: -1.0}, -np.inf, 0.0))                   # s <= x
        rows.append(({K + i: 1.0, S + i: -1.0}, -np.inf, 0.0))                   # k <= s
        rows.append(({F + i: 1.0, G + i: 1.0, X + i: -1.0, S + i: 1.0}, -np.inf, 0.0))  # f + g <= x - s
    for p in POSITIONS:
        idx = [i for i, c in enumerate(pool) if c.position == p]
        rows.append(({X + i: 1.0 for i in idx}, SQUAD_SIZE[p], SQUAD_SIZE[p]))
        rows.append(({S + i: 1.0 for i in idx}, XI_MIN[p], XI_MAX[p]))
    rows.append(({S + i: 1.0 for i in range(n)}, XI_SIZE, XI_SIZE))
    for var in (K, F, G):
        rows.append(({var + i: 1.0 for i in range(n)}, 1.0, 1.0))
    rows.append(({X + i: float(cost[i]) for i in range(n)}, -np.inf, float(budget)))
    for club in clubs:
        idx = [i for i, c in enumerate(pool) if c.team == club]
        if len(idx) > max_per_club:
            rows.append(({X + i: 1.0 for i in idx}, -np.inf, float(max_per_club)))

    a = lil_matrix((len(rows), 5 * n))
    lb = np.empty(len(rows))
    ub = np.empty(len(rows))
    for r, (coefs, lo, hi) in enumerate(rows):
        for j, v in coefs.items():
            a[r, j] = v
        lb[r], ub[r] = lo, hi

    # Maximise sum(s*p) + sum(k*p) + w_rest*bench + (w1 - w_rest)*f*p + (w2 - w_rest)*g*p
    w1, w2, w_rest = BENCH_WEIGHTS
    w_bench = np.where(is_gk, BENCH_GK_WEIGHT, w_rest)
    gain = np.concatenate([w_bench * points,               # x
                           (1.0 - w_bench) * points,       # s: from bench weight to 1
                           points,                         # captain counts twice
                           (w1 - w_rest) * points,
                           (w2 - w_rest) * points])
    upper = np.ones(5 * n)
    upper[F:F + n][is_gk] = 0                              # subs 1 and 2 are outfielders
    upper[G:G + n][is_gk] = 0
    res = milp(-gain, constraints=LinearConstraint(a.tocsr(), lb, ub),
               integrality=np.ones(5 * n), bounds=Bounds(0, upper),
               options={"mip_rel_gap": 0.0, "time_limit": time_limit, "presolve": True})
    if res.x is None:
        raise InfeasibleError(f"No valid squad: {res.message}")

    x = np.round(res.x[X:X + n]).astype(int)
    s = np.round(res.x[S:S + n]).astype(int)
    starters = order_starters(pool[i] for i in range(n) if s[i])
    bench = order_bench(pool[i] for i in range(n) if x[i] and not s[i])
    captain, vice = pick_captains(starters)
    return Selection(starters=starters, bench=bench, captain=captain, vice_captain=vice)


# ------------------------------------------------------------------ validation

def validate_squad(sel: Selection, budget: Optional[int] = DEFAULT_BUDGET,
                   max_per_club: Optional[int] = DEFAULT_MAX_PER_CLUB,
                   full_squad: bool = True) -> List[str]:
    """Constraint violations of a selection (empty list = valid). Used by tests."""
    problems: List[str] = []
    xi = sel.starters
    if len(xi) != XI_SIZE:
        problems.append(f"XI has {len(xi)} players")
    for p in POSITIONS:
        k = sum(c.position == p for c in xi)
        if not XI_MIN[p] <= k <= XI_MAX[p]:
            problems.append(f"XI has {k} {p}")
    squad = list(xi) + list(sel.bench)
    if len({c.id for c in squad}) != len(squad):
        problems.append("Duplicate players")
    if full_squad:
        for p in POSITIONS:
            k = sum(c.position == p for c in squad)
            if k != SQUAD_SIZE[p]:
                problems.append(f"Squad has {k} {p}")
    if budget is not None and sum(c.cost for c in squad) > budget:
        problems.append(f"Over budget: {sum(c.cost for c in squad)} > {budget}")
    if max_per_club is not None:
        for club in {c.team for c in squad}:
            k = sum(c.team == club for c in squad)
            if k > max_per_club:
                problems.append(f"{k} players from {club}")
    if sel.captain not in xi or (sel.vice_captain is not None and sel.vice_captain not in xi):
        problems.append("Captain/vice not in the XI")
    return problems


# ------------------------------------------------------------------ scoring: auto-subs

FORMATION_MIN: Dict[str, int] = {"DEF": 3, "MID": 2, "FWD": 1}


def _counts(players: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {p: 0 for p in POSITIONS}
    for pl in players:
        counts[pl["position"]] += 1
    return counts


def apply_auto_subs(starters: List[Dict[str, Any]], bench: List[Dict[str, Any]]
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]:
    """FPL automatic substitutions. Each player is a dict with `name`, `position` and
    `played`: True, False (his matches are over and he got no minutes) or None
    (undecided). A starter who didn't play is replaced by the first bench player in
    order (goalkeeper for goalkeeper) who played and keeps 3+ DEF, 2+ MID and 1+ FWD;
    an undecided bench player blocks the ones behind him. Returns (final XI, final bench
    in priority order, [(out, in) names]). A replaced starter takes his replacement's
    bench slot, as the FPL app shows it."""
    xi, slots = list(starters), list(bench)
    subs: List[Tuple[str, str]] = []
    used = set()
    for out in [p for p in starters if p["played"] is False]:
        for i, cand in enumerate(slots):
            if i in used or (cand["position"] == "GK") != (out["position"] == "GK"):
                continue
            counts = _counts(xi)
            counts[out["position"]] -= 1
            counts[cand["position"]] += 1
            if any(counts[p] < n for p, n in FORMATION_MIN.items()):
                continue
            if cand["played"] is None:
                break                      # may still play: no sub until he's decided
            if cand["played"] is False:
                continue
            xi[xi.index(out)] = cand
            slots[i] = out
            used.add(i)
            subs.append((out["name"], cand["name"]))
            break
    return xi, slots, subs
