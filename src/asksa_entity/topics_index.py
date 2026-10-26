from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .text_normalize import normalize_phrase


@dataclass
class TopicIndex:
    topic_norm_full: Dict[str, str]
    topic_tokens: Dict[str, Set[str]]
    token_df: Dict[str, int]
    token_to_topics: Dict[str, Set[str]]
    idf: Dict[str, float]

    token_min_len_for_index: int
    topic_fuzzy_min_words: int
    topic_rare_df_threshold: int
    topic_fuzzy_min_overlap: int
    topic_fuzzy_score_threshold: float


def tokenize_for_index(s: str, token_min_len_for_index: int) -> List[str]:
    s = normalize_phrase(s)
    return [t for t in s.split() if len(t) >= token_min_len_for_index]


def build_topic_index(
    topics_df: pd.DataFrame,
    token_min_len_for_index: int = 3,
    topic_fuzzy_min_words: int = 4,
    topic_rare_df_threshold: int = 3,
    topic_fuzzy_min_overlap: int = 2,
    topic_fuzzy_score_threshold: float = 0.75,
) -> TopicIndex:
    topic_canon_list = (
        topics_df["topic_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    topic_norm_full: Dict[str, str] = {}
    topic_tokens: Dict[str, Set[str]] = {}
    token_df: Dict[str, int] = {}
    token_to_topics: Dict[str, Set[str]] = {}

    for canon in topic_canon_list:
        n = normalize_phrase(canon)
        if not n:
            continue
        toks = set(tokenize_for_index(canon, token_min_len_for_index))
        if not toks:
            continue
        topic_norm_full[canon] = n
        topic_tokens[canon] = toks
        for tkn in toks:
            token_df[tkn] = token_df.get(tkn, 0) + 1
            token_to_topics.setdefault(tkn, set()).add(canon)

    N_topics = max(1, len(topic_tokens))
    idf = {
        t: float(np.log((N_topics + 1) / (df + 1)) + 1.0)
        for t, df in token_df.items()
    }

    return TopicIndex(
        topic_norm_full=topic_norm_full,
        topic_tokens=topic_tokens,
        token_df=token_df,
        token_to_topics=token_to_topics,
        idf=idf,
        token_min_len_for_index=token_min_len_for_index,
        topic_fuzzy_min_words=topic_fuzzy_min_words,
        topic_rare_df_threshold=topic_rare_df_threshold,
        topic_fuzzy_min_overlap=topic_fuzzy_min_overlap,
        topic_fuzzy_score_threshold=topic_fuzzy_score_threshold,
    )


def _topic_is_long_or_rare(index: TopicIndex, canon: str) -> bool:
    toks = index.topic_tokens.get(canon, set())
    if not toks:
        return False
    if (
        len(index.topic_norm_full.get(canon, "").split())
        >= index.topic_fuzzy_min_words
    ):
        return True
    min_df = min(index.token_df.get(t, 999999) for t in toks)
    return min_df <= index.topic_rare_df_threshold


def _score_topic_overlap(
    index: TopicIndex, msg_toks: Set[str], canon: str
) -> Tuple[float, int]:
    toks = index.topic_tokens.get(canon, set())
    inter = msg_toks & toks
    if not inter:
        return 0.0, 0
    num = sum(index.idf.get(t, 1.0) for t in inter)
    den = sum(index.idf.get(t, 1.0) for t in toks) + 1e-9
    return float(num / den), len(inter)


def fuzzy_topics_for_message(
    index: TopicIndex, msg_phrase_norm: str
) -> List[str]:
    msg_toks = set(
        [
            t
            for t in msg_phrase_norm.split()
            if len(t) >= index.token_min_len_for_index
        ]
    )
    if not msg_toks:
        return []

    cand_pool: Set[str] = set()
    for tkn in msg_toks:
        cand_pool |= index.token_to_topics.get(tkn, set())

    scored = []
    for canon in cand_pool:
        if not _topic_is_long_or_rare(index, canon):
            continue
        sc, ov = _score_topic_overlap(index, msg_toks, canon)
        if (
            ov >= index.topic_fuzzy_min_overlap
            and sc >= index.topic_fuzzy_score_threshold
        ):
            scored.append((sc, ov, canon))

    scored.sort(reverse=True)
    return [c for _, _, c in scored[:10]]
