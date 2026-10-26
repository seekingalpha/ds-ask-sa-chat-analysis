from __future__ import annotations

import re
from typing import Iterable, List, Tuple


class AhoPhraseMatcher:
    """
    Aho–Corasick phrase matcher if 'pyahocorasick' is installed,
    else fallback to exact n-gram matching.
    """

    def __init__(self, phrases: Iterable[str], max_ngram_fallback: int = 6):
        self.phrases = sorted(set([p for p in phrases if p]))
        self.max_ngram_fallback = max_ngram_fallback
        self.use_ac = False
        self.ac = None

        try:
            import ahocorasick  # type: ignore

            A = ahocorasick.Automaton()
            for p in self.phrases:
                A.add_word(p, p)
            A.make_automaton()
            self.use_ac = True
            self.ac = A
        except Exception:
            self.use_ac = False
            self.ac = None
            self.phrase_set = set(self.phrases)
            self.max_tokens = max(
                (len(p.split()) for p in self.phrases), default=1
            )

    def find(self, msg_norm: str) -> List[Tuple[int, int, str]]:
        out: List[Tuple[int, int, str]] = []
        if not msg_norm:
            return out

        if self.use_ac and self.ac is not None:
            for end_idx, p in self.ac.iter(msg_norm):
                start_idx = end_idx - len(p) + 1
                if start_idx > 0 and msg_norm[start_idx - 1].isalnum():
                    continue
                if (
                    end_idx + 1 < len(msg_norm)
                    and msg_norm[end_idx + 1].isalnum()
                ):
                    continue
                out.append((start_idx, end_idx + 1, p))
            return out

        words, spans = [], []
        for m in re.finditer(r"[a-z0-9]+", msg_norm):
            words.append(m.group(0))
            spans.append((m.start(), m.end()))
        max_n = min(self.max_tokens, self.max_ngram_fallback)

        for i in range(len(words)):
            for n in range(1, max_n + 1):
                j = i + n
                if j > len(words):
                    break
                phrase = " ".join(words[i:j])
                if phrase in self.phrase_set:
                    out.append((spans[i][0], spans[j - 1][1], phrase))
        return out
