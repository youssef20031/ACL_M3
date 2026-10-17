"""PlayerIndex mention spans, and the offset-tracking name normalization behind them."""
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypothesis import given, settings, strategies as st

from services.player_index import PlayerIndex
from utils.text import normalize_name, normalize_name_with_offsets

ROWS = [
    {"player_name": name, "position": position, "team_name": team, "is_current": current,
     "career_points": points}
    for name, position, team, current, points in [
        ("Conor Coventry", "MID", "West Ham", True, 40),
        ("Pierre Lees-Melou", "MID", "Norwich", False, 50),
        ("Martin Ødegaard", "MID", "Arsenal", True, 700),
        ("N'Golo Kanté", "MID", "Chelsea", False, 400),
        ("Mohamed Salah", "MID", "Liverpool", True, 1500),
        ("Erling Haaland", "FWD", "Man City", True, 1000),
    ]
]


class TestMentionSpans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = PlayerIndex(ROWS)

    def spans(self, text):
        """(the text that named the player, player, named in full) for each mention."""
        return [(text[m.start:m.end], m.player_name, m.full_name) for m in self.index.mention_spans(text)]

    def test_spans_index_the_original_text(self):
        # Accents, apostrophes and brackets don't shift the offsets
        self.assertEqual(self.spans("Is Martin Ødegaard (Arsenal) better than N'Golo Kanté?"), [
            ("Martin Ødegaard", "Martin Ødegaard", True),
            ("N'Golo Kanté", "N'Golo Kanté", True),
        ])

    def test_single_words_and_typos_are_not_full_names(self):
        self.assertEqual(self.spans("Salah or Haland?"), [
            ("Salah", "Mohamed Salah", False),
            ("Haland", "Erling Haaland", False),
        ])

    def test_a_player_named_twice_is_listed_twice(self):
        text = "Salah's form: is Mohamed Salah worth it?"
        self.assertEqual(self.spans(text), [
            ("Salah", "Mohamed Salah", False),
            ("Mohamed Salah", "Mohamed Salah", True),
        ])
        self.assertEqual(self.index.mentions(text), ["Mohamed Salah"])

    def test_mentions_are_unchanged(self):
        # The index knows nothing about clubs, so mentions() still reads club words as
        # players; the entity extractor uses the spans to drop them
        text = "Coventry City vs Leeds United"
        self.assertEqual(self.index.mentions(text), ["Conor Coventry", "Pierre Lees-Melou"])
        self.assertEqual(self.spans(text), [
            ("Coventry", "Conor Coventry", False),
            ("Leeds", "Pierre Lees-Melou", False),
        ])
        self.assertEqual(self.index.mentions("Haaland, Salah and Kanté", limit=2),
                         ["Erling Haaland", "Mohamed Salah"])
        self.assertEqual(self.index.mention_spans("?!"), [])


class TestNormalizeNameWithOffsets(unittest.TestCase):
    def test_maps_a_match_back_to_the_original(self):
        text = "Ødegaard (Arsenal)"
        normalized, offsets = normalize_name_with_offsets(text)
        self.assertEqual(normalized, "odegaard arsenal")
        start, end = normalized.index("arsenal"), len(normalized)
        self.assertEqual(text[offsets[start]:offsets[end - 1] + 1], "Arsenal")

    def test_letters_that_fold_to_several(self):
        # "æ" folds to two letters, and both point at it
        normalized, offsets = normalize_name_with_offsets("Bæk İlkay")
        self.assertEqual(normalized, "baek ilkay")
        self.assertEqual(offsets, [0, 1, 1, 2, 3, 4, 5, 6, 7, 8])

    @settings(max_examples=500, deadline=None)
    @given(st.text())
    def test_same_text_as_normalize_name(self, text):
        normalized, offsets = normalize_name_with_offsets(text)
        self.assertEqual(normalized, normalize_name(text))
        self.assertEqual(len(offsets), len(normalized))
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(0 <= i < len(text) for i in offsets))


if __name__ == "__main__":
    unittest.main()
