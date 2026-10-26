"""Filter for obvious ticker false positives before the LLM sees them.

Ticker candidates come from matching every word of a message against all
tickers, so ordinary words ("news" -> NE.WS, "hold" -> HOLD, "or" -> OR)
become candidates. The LLM usually rejects them, but each one costs an LLM
call and sometimes slips through. This module drops only cases with no
ticker signal at all; anything ambiguous is left to the LLM.

A ticker-like token is NOISE when one of these holds:
  url: the token is part of a URL.
  word_for_dotted_ticker: every matched ticker has a separator (NE.WS,
      BF.B) but the token has none and is not written in capitals.
  common_word: the token is a common word (COMMON_WORDS) written in lower
      or Title case (or in an ALL-CAPS message), and none of the ticker
      signals below is present.

Ticker signals (keep the candidate):
  - "$" right before the token, or the token in parentheses: $NOW, (HOLD)
  - capitals in a message that is not all capitals: "buy NOW or later?"
  - a list/comparison next to the token: "cat, fix, vrt", "ge vs ph",
    one ticker per line
  - the token is the whole message: "eat"
"""

from __future__ import annotations

import re
from typing import Optional

# Lowercase words that are also tickers but almost never mean the ticker in
# running text. Brand-like words users may write in lowercase for the
# company (meta, snow, coin, cat, net, ...) are deliberately NOT here.
COMMON_WORDS = frozenset("""
    a an the and or but if so as at by for from in into of on onto to up down out off over under
    with without about above after before between through per via vs than then too very also just
    only even still yet again ever never now here there where when why how what which who whom whose
    this that these those it its is am are was were be been being do does did done have has had
    can could will would shall should may might must not no yes all any some each every both few
    many much more most less least other another such own same
    i me my mine we us our ours you your he him his she her they them their
    one two three four five six ten first second next last new old good best better great big small
    high low top long full free real true well open close plus sure fast safe live life love like
    want need help find give get got go going make take look see know think say tell show use try
    keep hold add sell pay play run win work home team care info news fund funds week weeks month
    year day days time mid cap rate rates cash money deal plan idea ideas tips list lists rank today
    ok okay please thanks thank hi hello hey fix gold silver oil gas usa uk eu
    de la el en los las un una con para que por del al es
    der die das und ein eine mit von zu ist im
    het een van met dit dat wat voor op
    """.split())

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SEPARATORS = ".-/:"
_LIST_CHARS = ",;\n\t|"
_VS_BEFORE_RE = re.compile(r"\b(?:vs|v|versus)\.?\s*$", re.IGNORECASE)
_VS_AFTER_RE = re.compile(r"^\s*(?:vs|v|versus)\b", re.IGNORECASE)


def url_spans(text: str) -> list:
    return [m.span() for m in _URL_RE.finditer(text)]


def is_all_caps_message(text: str) -> bool:
    """True if letters are mostly capitals: then capitals are no ticker signal."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.7


def _has_ticker_signal(
    token: str, start: int, end: int, text: str, all_caps: bool
) -> bool:
    before = text[:start]
    after = text[end:]
    if before.endswith("$") or (
        before.rstrip().endswith("(") and after.lstrip().startswith(")")
    ):
        return True
    if token.isupper() and not all_caps:
        return True
    prev_char = before.rstrip(" ")[-1:]
    next_char = after.lstrip(" ")[:1]
    if (prev_char and prev_char in _LIST_CHARS) or (
        next_char and next_char in _LIST_CHARS
    ):
        return True
    if _VS_BEFORE_RE.search(before) or _VS_AFTER_RE.search(after):
        return True
    if text.strip(" \t\n?!.$") == token:
        return True
    return False


def ticker_noise_reason(
    token: str,
    start: int,
    end: int,
    text: str,
    compact: str,
    bundle: str,
    urls: list,
    all_caps: bool,
) -> Optional[str]:
    """Rule name if this ticker match is noise (see module docstring), else None."""
    if any(s <= start < e for s, e in urls):
        return "url"

    members = bundle.split("|")
    has_sep = any(ch in token for ch in _SEPARATORS)
    if (
        not has_sep
        and not token.isupper()
        and all(any(ch in m for ch in _SEPARATORS) for m in members)
    ):
        return "word_for_dotted_ticker"

    if compact in COMMON_WORDS and not _has_ticker_signal(
        token, start, end, text, all_caps
    ):
        return "common_word"
    return None
