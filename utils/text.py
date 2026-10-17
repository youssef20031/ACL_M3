"""
Text normalization shared by the data loader, player search and name resolution.
"""
import re
import unicodedata
from typing import List, Tuple

# Letters that Unicode NFKD does not decompose into an ASCII base + combining mark.
_SPECIAL_LETTERS = str.maketrans({
    "ø": "o", "Ø": "o",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
    "ß": "ss",
    "đ": "d", "Đ": "d",
    "ð": "d", "Ð": "d",
    "ł": "l", "Ł": "l",
    "ı": "i",
    "þ": "th", "Þ": "th",
})

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    """Strip accents and lowercase, keeping punctuation and spaces."""
    text = text.translate(_SPECIAL_LETTERS)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return text.lower()


def normalize_name(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    "Martin Ødegaard" -> "martin odegaard", "N'Golo Kanté" -> "n golo kante".
    """
    if not text:
        return ""
    return _NON_ALNUM.sub(" ", _fold(text)).strip()


def normalize_name_with_offsets(text: str) -> Tuple[str, List[int]]:
    """normalize_name(text), plus the index in `text` that each of its characters comes from.

    So a match in the normalized text can be mapped back: in "Ødegaard (Arsenal)", the
    normalized "arsenal" starts at 9 and offsets[9] == 10, the "A". A space stands for a run
    of punctuation/whitespace and points at its first character.
    """
    chars: List[str] = []
    offsets: List[int] = []
    # Folding one character at a time gives the same result as folding the whole text: NFKD
    # works per character (reordering only combining marks, which are dropped).
    for i, original in enumerate(text or ""):
        for c in _fold(original):
            if not _NON_ALNUM.match(c):
                chars.append(c)
                offsets.append(i)
            elif chars and chars[-1] != " ":
                chars.append(" ")
                offsets.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets
