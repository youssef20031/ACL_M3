"""
Text normalization shared by the data loader, player search and name resolution.
"""
import re
import unicodedata

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


def normalize_name(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    "Martin Ødegaard" -> "martin odegaard", "N'Golo Kanté" -> "n golo kante".
    """
    if not text:
        return ""
    text = text.translate(_SPECIAL_LETTERS)
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return _NON_ALNUM.sub(" ", text.lower()).strip()
