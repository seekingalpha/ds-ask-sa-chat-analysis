from __future__ import annotations

import re
import unicodedata
from typing import Iterator, Set, Tuple

_keep_alnum_space = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")
_zero_width_re = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def normalize_base(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _zero_width_re.sub("", s)
    s = s.lower()
    s = (
        s.replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    s = _ws.sub(" ", s).strip()
    return s


def normalize_tokens(s: str) -> str:
    """
    For message tokenization + tickers:
    - lowercase
    - unify common punctuation to spaces
    - remove non-alnum (keep spaces)
    - IMPORTANT: replace apostrophe with space so "I'm" -> "i m"
    """
    s = normalize_base(s)
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[(){}\[\],;:]", " ", s)
    s = s.replace("'", " ")
    s = _keep_alnum_space.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s


def normalize_phrase(s: str) -> str:
    """For phrase matching: same as tokens for consistency."""
    return normalize_tokens(s)


_ticker_like_re = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9\.\-\/]{0,14}\b")
_apostrophe_tail_re = re.compile(r"'(m|s|re|ve|d|ll)\b", flags=re.IGNORECASE)


def ticker_compact(s: str) -> str:
    """
    Compact ticker:
    - lower
    - remove all non-alnum
    BRK.B / BRK-B / BRKB -> brkb
    """
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def iter_ticker_like_matches(original: str) -> Iterator[Tuple[str, int, int]]:
    """Like iter_ticker_like_tokens, with (token, start, end) positions in ORIGINAL."""
    s = "" if original is None else str(original)
    for m in _ticker_like_re.finditer(s):
        yield m.group(0), m.start(), m.end()


def iter_ticker_like_tokens(original: str) -> Iterator[str]:
    """Yield ticker-like tokens (keeps '.', '-', '/', etc.) from the ORIGINAL message."""
    s = "" if original is None else str(original)
    for m in _ticker_like_re.finditer(s):
        yield m.group(0)


def contraction_tails_in_message(original: str) -> Set[str]:
    """Detect tails that occur after apostrophe in original text: I'm -> 'm."""
    s = "" if original is None else str(original)
    return set(m.group(1).lower() for m in _apostrophe_tail_re.finditer(s))
