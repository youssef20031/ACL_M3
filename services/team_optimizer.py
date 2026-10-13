"""
Team-of-the-week selection: pure functions, no Neo4j or file IO.

  best_xi(pool)       best valid starting XI by points (no budget, no club limit), like
                      FPL's Dream Team. Exact greedy: take each position's minimum, then
                      fill the remaining slots with the best players left under the caps.
  best_squad(pool)    full 15-man FPL squad (2 GK, 5 DEF, 5 MID, 3 FWD) within a budget and
                      a per-club limit that maximises the points of the XI picked from it.
                      Exact, via scipy's MILP solver (HiGHS), in two stages:
                        1. maximise the XI's points;
                        2. keeping that XI total, make the 4 bench players as cheap as
                           possible (ties: more bench points).

Costs are integers in FPL tenths (156 = 15.6m). Ties are deterministic: the pool is
sorted before solving and every ordering ends on the candidate id.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

POSITIONS: Tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
SQUAD_SIZE: Dict[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN: Dict[str, int] = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX: Dict[str, int] = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
XI_SIZE = 11
DEFAULT_BUDGET = 1000      # tenths: 100.0m
DEFAULT_MAX_PER_CLUB = 3


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
    """Best 15-man squad within budget/club limits, by the points of its starting XI.

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

    # Variables: x_0..x_{n-1} (in squad), s_0..s_{n-1} (starts). Rows built below.
    rows: List[Tuple[Dict[int, float], float, float]] = []
    for i in range(n):                                    # s_i <= x_i
        rows.append(({n + i: 1.0, i: -1.0}, -np.inf, 0.0))
    for p in POSITIONS:
        idx = [i for i, c in enumerate(pool) if c.position == p]
        rows.append(({i: 1.0 for i in idx}, SQUAD_SIZE[p], SQUAD_SIZE[p]))
        rows.append(({n + i: 1.0 for i in idx}, XI_MIN[p], XI_MAX[p]))
    rows.append(({n + i: 1.0 for i in range(n)}, XI_SIZE, XI_SIZE))
    rows.append(({i: float(cost[i]) for i in range(n)}, -np.inf, float(budget)))
    for club in clubs:
        idx = [i for i, c in enumerate(pool) if c.team == club]
        if len(idx) > max_per_club:
            rows.append(({i: 1.0 for i in idx}, -np.inf, float(max_per_club)))

    def build(extra: List[Tuple[Dict[int, float], float, float]]) -> LinearConstraint:
        all_rows = rows + extra
        a = lil_matrix((len(all_rows), 2 * n))
        lb = np.empty(len(all_rows))
        ub = np.empty(len(all_rows))
        for r, (coefs, lo, hi) in enumerate(all_rows):
            for j, v in coefs.items():
                a[r, j] = v
            lb[r], ub[r] = lo, hi
        return LinearConstraint(a.tocsr(), lb, ub)

    integrality = np.ones(2 * n)
    bounds = Bounds(0, 1)
    options = {"mip_rel_gap": 0.0, "time_limit": time_limit, "presolve": True}

    # Stage 1: maximise the starting XI's points
    c1 = np.concatenate([np.zeros(n), -points])
    res1 = milp(c1, constraints=build([]), integrality=integrality, bounds=bounds, options=options)
    if res1.x is None:
        raise InfeasibleError(f"No valid squad: {res1.message}")
    best_xi_points = float(points @ np.round(res1.x[n:]))

    # Stage 2: same XI total, cheapest bench (then most bench points)
    tol = 1e-6 * max(1.0, abs(best_xi_points))
    keep_xi = ({n + i: float(points[i]) for i in range(n)}, best_xi_points - tol, np.inf)
    delta = 0.5 / (4 * max(float(points.max(initial=0.0)), 1.0) + 1)
    bench_cost = cost - delta * points                     # per bench player
    c2 = np.concatenate([bench_cost, -bench_cost])         # sum over x - s
    res2 = milp(c2, constraints=build([keep_xi]), integrality=integrality, bounds=bounds,
                options=options)
    sol = res2.x if res2.x is not None else res1.x

    x = np.round(sol[:n]).astype(int)
    s = np.round(sol[n:]).astype(int)
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
