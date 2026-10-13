"""Tests for services.team_optimizer (pure; no Neo4j)."""
import random
import unittest

from services import team_optimizer as opt
from services.team_optimizer import Candidate


def make_pool(seed: int, per_club: int = 6, clubs: int = 10):
    """Random pool with every position represented at every club."""
    rng = random.Random(seed)
    positions = ["GK", "DEF", "DEF", "MID", "MID", "FWD"]
    pool = []
    for c in range(clubs):
        for k in range(per_club):
            pos = positions[k % len(positions)]
            pool.append(Candidate(id=f"P{c:02d}-{k:02d}", position=pos, team=f"Club{c:02d}",
                                  points=round(rng.uniform(0.5, 9.0), 2), cost=rng.randint(40, 110)))
    return pool


def obvious_pool():
    """Stars from distinct clubs plus a cheap fodder tier; the optimum is clear."""
    pool = [
        Candidate("GK star", "GK", "A", 6.0, 55),
        Candidate("DEF star 1", "DEF", "B", 6.0, 50),
        Candidate("DEF star 2", "DEF", "C", 5.5, 50),
        Candidate("DEF star 3", "DEF", "D", 5.0, 45),
        Candidate("MID star 1", "MID", "E", 9.0, 110),
        Candidate("MID star 2", "MID", "F", 8.0, 90),
        Candidate("MID star 3", "MID", "G", 7.0, 90),
        Candidate("MID star 4", "MID", "H", 6.5, 80),
        Candidate("FWD star 1", "FWD", "I", 8.5, 120),
        Candidate("FWD star 2", "FWD", "J", 7.5, 90),
        Candidate("FWD star 3", "FWD", "K", 2.0, 60),
    ]
    # Stars cost 84.0m; cheap fodder from many clubs fills the bench (14.1m)
    fodder_clubs = [f"Z{i}" for i in range(12)]
    i = 0
    for pos, n in (("GK", 3), ("DEF", 4), ("MID", 3), ("FWD", 3)):
        for k in range(n):
            pool.append(Candidate(f"{pos} fodder {k}", pos, fodder_clubs[i % 12], 0.5 + 0.1 * k, 35 + k))
            i += 1
    return pool


class TestBestSquad(unittest.TestCase):
    def assertValid(self, sel, budget=opt.DEFAULT_BUDGET, max_per_club=opt.DEFAULT_MAX_PER_CLUB):
        self.assertEqual(opt.validate_squad(sel, budget, max_per_club), [])
        self.assertEqual(len(sel.starters), 11)
        self.assertEqual(len(sel.bench), 4)

    def test_constraints_hold_on_random_pools(self):
        for seed in range(6):
            pool = make_pool(seed)
            for budget in (1000, 850):
                sel = opt.best_squad(pool, budget=budget)
                self.assertValid(sel, budget=budget)

    def test_club_limit_binds(self):
        # One club holds all the best players; only 3 may be picked
        pool = [Candidate(f"Star{k}", pos, "Big", 10.0 - k, 50)
                for k, pos in enumerate(["GK", "DEF", "DEF", "MID", "MID", "FWD", "FWD"])]
        pool += make_pool(1, clubs=8)
        sel = opt.best_squad(pool)
        self.assertValid(sel)
        squad = sel.starters + sel.bench
        self.assertEqual(sum(c.team == "Big" for c in squad), 3)

    def test_obvious_optimum(self):
        sel = opt.best_squad(obvious_pool())
        self.assertValid(sel)
        names = {c.id for c in sel.starters}
        expected = {"GK star", "DEF star 1", "DEF star 2", "DEF star 3", "MID star 1", "MID star 2",
                    "MID star 3", "MID star 4", "FWD star 1", "FWD star 2"}
        self.assertTrue(expected <= names, names)
        # The 11th starter is FWD star 3 (2.0), better than any fodder (<1.0)
        self.assertIn("FWD star 3", names)
        self.assertEqual(sel.formation, "3-4-3")
        self.assertEqual(sel.captain.id, "MID star 1")
        self.assertEqual(sel.vice_captain.id, "FWD star 1")
        # Bench is the cheapest fodder that completes the squad
        self.assertTrue(all("fodder" in c.id for c in sel.bench))
        self.assertEqual(sel.bench[0].position, "GK")

    def test_bench_is_cheapest_possible(self):
        pool = obvious_pool() + [Candidate("GK pricey backup", "GK", "Z5", 0.9, 50)]
        sel = opt.best_squad(pool)
        bench_ids = {c.id for c in sel.bench}
        self.assertNotIn("GK pricey backup", bench_ids)
        self.assertIn("GK fodder 0", bench_ids)   # cost 35, cheapest goalkeeper
        self.assertEqual(sum(c.cost for c in sel.bench), 35 + 35 + 36 + 35)

    def test_budget_forces_cheaper_xi(self):
        unlimited = opt.best_squad(obvious_pool(), budget=10_000)
        tight = opt.best_squad(obvious_pool(), budget=900)
        self.assertValid(tight, budget=900)
        self.assertLess(sum(c.points for c in tight.starters), sum(c.points for c in unlimited.starters))

    def test_deterministic_and_order_independent(self):
        pool = make_pool(3)
        first = opt.best_squad(pool)
        shuffled = list(pool)
        random.Random(9).shuffle(shuffled)
        second = opt.best_squad(shuffled)
        self.assertEqual([c.id for c in first.starters + first.bench],
                         [c.id for c in second.starters + second.bench])
        self.assertEqual((first.captain.id, first.vice_captain.id),
                         (second.captain.id, second.vice_captain.id))

    def test_starter_and_bench_order(self):
        sel = opt.best_squad(make_pool(4))
        order = [opt.POSITIONS.index(c.position) for c in sel.starters]
        self.assertEqual(order, sorted(order))
        for p in opt.POSITIONS:
            pts = [c.points for c in sel.starters if c.position == p]
            self.assertEqual(pts, sorted(pts, reverse=True))

    def test_infeasible(self):
        with self.assertRaises(opt.InfeasibleError):
            opt.best_squad([c for c in make_pool(0) if c.position != "GK"])
        with self.assertRaises(opt.InfeasibleError):
            opt.best_squad(make_pool(0), budget=100)


class TestBestXI(unittest.TestCase):
    def test_valid_formation_and_optimum(self):
        pool = [
            Candidate("G1", "GK", "A", 2), Candidate("G2", "GK", "A", 12),
            *[Candidate(f"D{k}", "DEF", "A", 10 - k) for k in range(7)],
            *[Candidate(f"M{k}", "MID", "A", 9 - k) for k in range(7)],
            *[Candidate(f"F{k}", "FWD", "A", 15 - 6 * k) for k in range(4)],
        ]
        sel = opt.best_xi(pool)
        self.assertEqual(opt.validate_squad(sel, budget=None, max_per_club=None, full_squad=False), [])
        ids = [c.id for c in sel.starters]
        self.assertEqual(ids[0], "G2")
        # No club limit, no budget: just the best under the formation caps
        self.assertEqual(sorted(ids), sorted(["G2", "D0", "D1", "D2", "D3", "D4", "M0", "M1", "M2",
                                              "F0", "F1"]))
        self.assertEqual(sel.formation, "5-3-2")
        self.assertEqual(sel.captain.id, "F0")
        self.assertEqual(sel.vice_captain.id, "G2")
        self.assertEqual(sel.bench, [])

    def test_minimums_respected_even_when_low_scoring(self):
        pool = [Candidate("G", "GK", "A", 0), Candidate("F", "FWD", "A", 0)]
        pool += [Candidate(f"D{k}", "DEF", "A", 10) for k in range(6)]
        pool += [Candidate(f"M{k}", "MID", "A", 10) for k in range(6)]
        sel = opt.best_xi(pool)
        self.assertEqual(opt.validate_squad(sel, None, None, full_squad=False), [])
        self.assertIn("F", [c.id for c in sel.starters])

    def test_ties_are_deterministic(self):
        pool = [Candidate(f"X{k}", pos, "A", 5) for k, pos in
                enumerate(["GK", "GK"] + ["DEF"] * 6 + ["MID"] * 6 + ["FWD"] * 4)]
        a = opt.best_xi(pool)
        b = opt.best_xi(list(reversed(pool)))
        self.assertEqual([c.id for c in a.starters], [c.id for c in b.starters])
        self.assertEqual(a.captain.id, b.captain.id)

    def test_infeasible(self):
        with self.assertRaises(opt.InfeasibleError):
            opt.best_xi([Candidate("G", "GK", "A", 1)])


if __name__ == "__main__":
    unittest.main()
