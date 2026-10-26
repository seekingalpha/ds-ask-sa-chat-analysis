from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .config import ExtractConfig
from .phrase_matcher import AhoPhraseMatcher
from .text_normalize import (
    contraction_tails_in_message,
    iter_ticker_like_matches,
    normalize_phrase,
    ticker_compact,
)
from .ticker_noise import is_all_caps_message, ticker_noise_reason, url_spans
from .topics_index import (
    TopicIndex,
    build_topic_index,
    fuzzy_topics_for_message,
)

PHRASE_DROP_WORDS = {
    "of",
    "the",
    "and",
    "for",
    "in",
    "on",
    "to",
    "a",
    "an",
    "with",
    "by",
    "from",
}


def extract_context(original: str, window: int = 60) -> str:
    orig = "" if original is None else str(original)
    return orig[: min(len(orig), window * 2)]


def generate_phrase_aliases(
    phrase_norm: str, make_bigrams: bool = True
) -> List[str]:
    toks = phrase_norm.split()
    if not toks:
        return []
    toks2 = [t for t in toks if t not in PHRASE_DROP_WORDS]
    out = {phrase_norm}
    if toks2:
        out.add(" ".join(toks2))
    if make_bigrams and len(toks2) >= 4:
        out.add(" ".join(toks2[:2]))
        out.add(" ".join(toks2[-2:]))
    return [x for x in out if len(x) >= 3]


def prep_phrases_with_unique_aliases(
    values: Iterable[str], min_len: int, make_bigrams: bool
) -> Dict[str, str]:
    alias_to_originals: Dict[str, List[str]] = {}
    for v in values:
        v0 = "" if v is None else str(v).strip()
        if not v0:
            continue
        p = normalize_phrase(v0)
        if not p:
            continue
        for a in generate_phrase_aliases(p, make_bigrams=make_bigrams):
            if len(a) >= min_len:
                alias_to_originals.setdefault(a, []).append(v0)

    out: Dict[str, str] = {}
    for a, originals in alias_to_originals.items():
        uniq = sorted(set(originals))
        if len(uniq) == 1:
            out[a] = uniq[0]
        else:
            for o in uniq:
                if a == normalize_phrase(o):
                    out[a] = o
                    break
    return out


def prep_tickers_v2(
    tickers: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    t = tickers.copy()
    t["ticker_id"] = t["ticker_id"].astype(str).str.upper().str.strip()
    t["company_name"] = t["company_name"].astype(str).str.strip()

    ticker_to_company = dict(zip(t["ticker_id"], t["company_name"]))

    compact_map: Dict[str, List[str]] = {}
    for tid in t["ticker_id"].tolist():
        c = ticker_compact(tid)
        if not c:
            continue
        compact_map.setdefault(c, []).append(tid)

    compact_to_bundle: Dict[str, str] = {}
    for c, tids in compact_map.items():
        tids2 = sorted(set(tids))
        compact_to_bundle[c] = tids2[0] if len(tids2) == 1 else "|".join(tids2)

    return compact_to_bundle, ticker_to_company


def prep_company_phrases(tickers: pd.DataFrame) -> Dict[str, str]:
    t = tickers.copy()
    t["ticker_id"] = t["ticker_id"].astype(str).str.upper().str.strip()
    t["company_name"] = t["company_name"].astype(str).str.strip()

    phrase_to_tickers: Dict[str, List[str]] = {}
    for tid, cname in zip(t["ticker_id"], t["company_name"]):
        p = normalize_phrase(cname)
        if p and len(p) >= 3:
            phrase_to_tickers.setdefault(p, []).append(tid)

    phrase_to_bundle: Dict[str, str] = {}
    for p, tids in phrase_to_tickers.items():
        tids2 = sorted(set(tids))
        phrase_to_bundle[p] = tids2[0] if len(tids2) == 1 else "|".join(tids2)
    return phrase_to_bundle


@dataclass
class EntityResources:
    cfg: ExtractConfig
    compact_to_bundle: Dict[str, str]
    ticker_to_company: Dict[str, str]
    company_phrase_to_bundle: Dict[str, str]

    sector_phrase_map: Dict[str, str]
    industry_phrase_map: Dict[str, str]
    topic_phrase_map: Dict[str, str]

    company_matcher: AhoPhraseMatcher
    sector_matcher: AhoPhraseMatcher
    industry_matcher: AhoPhraseMatcher
    topic_matcher: AhoPhraseMatcher

    topic_index: TopicIndex


@dataclass
class MatchCandidate:
    entity_type: str
    entity_id: str
    canonical: str
    matched_text_norm: str
    context: str
    match_source: str
    confidence: float = 0.50


def build_resources(
    tickers_df: pd.DataFrame,
    secind_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    cfg: Optional[ExtractConfig] = None,
) -> EntityResources:
    cfg = cfg or ExtractConfig()

    compact_to_bundle, ticker_to_company = prep_tickers_v2(tickers_df)
    company_phrase_to_bundle = prep_company_phrases(tickers_df)

    sector_phrase_map = prep_phrases_with_unique_aliases(
        secind_df["sector_name"].dropna().unique().tolist(),
        min_len=3,
        make_bigrams=False,
    )
    industry_phrase_map = prep_phrases_with_unique_aliases(
        secind_df["industry_name"].dropna().unique().tolist(),
        min_len=3,
        make_bigrams=False,
    )
    topic_phrase_map = prep_phrases_with_unique_aliases(
        topics_df["topic_name"].dropna().unique().tolist(),
        min_len=2,
        make_bigrams=True,
    )

    company_matcher = AhoPhraseMatcher(
        company_phrase_to_bundle.keys(),
        max_ngram_fallback=cfg.max_ngram_fallback,
    )
    sector_matcher = AhoPhraseMatcher(
        sector_phrase_map.keys(), max_ngram_fallback=cfg.max_ngram_fallback
    )
    industry_matcher = AhoPhraseMatcher(
        industry_phrase_map.keys(), max_ngram_fallback=cfg.max_ngram_fallback
    )
    topic_matcher = AhoPhraseMatcher(
        topic_phrase_map.keys(), max_ngram_fallback=cfg.max_ngram_fallback
    )

    topic_index = build_topic_index(
        topics_df,
        token_min_len_for_index=cfg.token_min_len_for_index,
        topic_fuzzy_min_words=cfg.topic_fuzzy_min_words,
        topic_rare_df_threshold=cfg.topic_rare_df_threshold,
        topic_fuzzy_min_overlap=cfg.topic_fuzzy_min_overlap,
        topic_fuzzy_score_threshold=cfg.topic_fuzzy_score_threshold,
    )

    return EntityResources(
        cfg=cfg,
        compact_to_bundle=compact_to_bundle,
        ticker_to_company=ticker_to_company,
        company_phrase_to_bundle=company_phrase_to_bundle,
        sector_phrase_map=sector_phrase_map,
        industry_phrase_map=industry_phrase_map,
        topic_phrase_map=topic_phrase_map,
        company_matcher=company_matcher,
        sector_matcher=sector_matcher,
        industry_matcher=industry_matcher,
        topic_matcher=topic_matcher,
        topic_index=topic_index,
    )


def extract_candidates_from_message(
    msg: str,
    R: EntityResources,
    dropped: Optional[List[Dict[str, Any]]] = None,
) -> List[MatchCandidate]:
    """Candidates for one message.

    dropped: if a list is given, ticker matches removed by the noise filter
    are appended to it (token, ticker, rule) for inspection.
    """
    original = "" if msg is None else str(msg)
    msg_phrase_norm = normalize_phrase(original)

    cands: List[MatchCandidate] = []

    # 1) tickers: compact match from ticker-like tokens in original
    tails = contraction_tails_in_message(original)
    urls = url_spans(original) if R.cfg.ticker_noise_filter else []
    all_caps = is_all_caps_message(original)
    for raw_token, start, end in iter_ticker_like_matches(original):
        c = ticker_compact(raw_token)
        if not c:
            continue
        if c in tails:
            continue
        bundle = R.compact_to_bundle.get(c)
        if not bundle:
            continue

        # avoid false positives for 1-char tickers unless EXACT single uppercase letter in message
        if len(c) == 1:
            if not (
                len(raw_token) == 1
                and raw_token.isalpha()
                and raw_token.isupper()
            ):
                continue

        if R.cfg.ticker_noise_filter:
            reason = ticker_noise_reason(
                raw_token, start, end, original, c, bundle, urls, all_caps
            )
            if reason:
                if dropped is not None:
                    dropped.append(
                        {"token": raw_token, "ticker": bundle, "rule": reason}
                    )
                continue

        canonical = R.ticker_to_company.get(bundle, bundle)
        cands.append(
            MatchCandidate(
                "ticker",
                bundle,
                canonical,
                c,
                extract_context(original, window=R.cfg.context_window),
                "ticker_compact",
                0.56 if "|" not in bundle else 0.52,
            )
        )

    # 2) company name phrases -> ticker bundle
    for _start, _end, phrase in R.company_matcher.find(msg_phrase_norm):
        bundle = R.company_phrase_to_bundle.get(phrase, "")
        if not bundle:
            continue
        canon = (
            R.ticker_to_company.get(bundle, phrase)
            if "|" not in bundle
            else phrase
        )
        cands.append(
            MatchCandidate(
                "ticker",
                bundle,
                canon,
                phrase,
                extract_context(original, R.cfg.context_window),
                "company_name",
                0.60,
            )
        )

    # 3) sector/industry/topic exact
    for _s, _e, phrase in R.sector_matcher.find(msg_phrase_norm):
        canon = R.sector_phrase_map.get(phrase, phrase)
        cands.append(
            MatchCandidate(
                "sector",
                canon,
                canon,
                phrase,
                extract_context(original, R.cfg.context_window),
                "phrase",
                0.60,
            )
        )

    for _s, _e, phrase in R.industry_matcher.find(msg_phrase_norm):
        canon = R.industry_phrase_map.get(phrase, phrase)
        cands.append(
            MatchCandidate(
                "industry",
                canon,
                canon,
                phrase,
                extract_context(original, R.cfg.context_window),
                "phrase",
                0.60,
            )
        )

    exact_topics: List[str] = []
    for _s, _e, phrase in R.topic_matcher.find(msg_phrase_norm):
        canon = R.topic_phrase_map.get(phrase, phrase)
        exact_topics.append(canon)
        cands.append(
            MatchCandidate(
                "topic",
                canon,
                canon,
                phrase,
                extract_context(original, R.cfg.context_window),
                "phrase",
                0.58,
            )
        )

    # 4) fuzzy topics only if no exact match in this message
    if not exact_topics:
        for canon in fuzzy_topics_for_message(R.topic_index, msg_phrase_norm):
            cands.append(
                MatchCandidate(
                    "topic",
                    canon,
                    canon,
                    matched_text_norm=R.topic_index.topic_norm_full.get(
                        canon, normalize_phrase(canon)
                    ),
                    context=extract_context(original, R.cfg.context_window),
                    match_source="topic_fuzzy_idf",
                    confidence=0.53,
                )
            )

    # 5) dedup
    uniq: Dict[Tuple[str, str, str, str], MatchCandidate] = {}
    for c in cands:
        k = (c.entity_type, c.entity_id, c.matched_text_norm, c.match_source)
        if k not in uniq or uniq[k].confidence < c.confidence:
            uniq[k] = c
    return list(uniq.values())


def iter_message_batches(
    df_messages: pd.DataFrame, batch_size: int
) -> Iterator[pd.DataFrame]:
    n = len(df_messages)
    for i in range(0, n, batch_size):
        yield df_messages.iloc[i : i + batch_size].copy()


def extract_raw_matches(
    messages_df: pd.DataFrame,
    R: EntityResources,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for b, df in enumerate(
        tqdm(
            iter_message_batches(messages_df, R.cfg.batch_size),
            desc="extract raw",
        )
    ):
        if max_batches is not None and b >= max_batches:
            break
        for r in df.itertuples(index=False):
            msg_text = str(r.msg)
            for c in extract_candidates_from_message(msg_text, R):
                rows.append(
                    {
                        "ts": r.ts,
                        "user_id": r.user_id,
                        "chat_id": r.chat_id,
                        "trace_id": r.trace_id,
                        "msg": msg_text,
                        "entity_type": c.entity_type,
                        "entity_id": c.entity_id,
                        "canonical": c.canonical,
                        "matched_text_norm": c.matched_text_norm,
                        "match_source": c.match_source,
                        "confidence": float(c.confidence),
                        "context": c.context,
                    }
                )
    return pd.DataFrame(rows)


def explain_ticker_noise(
    messages_df: pd.DataFrame, R: EntityResources
) -> pd.DataFrame:
    """Ticker matches the noise filter removes: one row per (message, token)."""
    rows: List[Dict[str, Any]] = []
    for r in messages_df.itertuples(index=False):
        dropped: List[Dict[str, Any]] = []
        extract_candidates_from_message(str(r.msg), R, dropped=dropped)
        for d in dropped:
            rows.append({"trace_id": r.trace_id, "msg": str(r.msg), **d})
    return pd.DataFrame(
        rows, columns=["trace_id", "msg", "token", "ticker", "rule"]
    )


CONTRACTION_TAILS = {"m", "s", "re", "ve", "d", "ll"}


def drop_contraction_artifacts(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extra safety net: if you still keep a 'ticker_symbol' path elsewhere.
    In this package, tickers use ticker_compact; keep for compatibility.
    """
    if raw_df.empty:
        return raw_df
    d = raw_df.copy()
    if "match_source" not in d.columns:
        return d
    is_ticker = d["entity_type"].eq("ticker")
    is_symbol = d["match_source"].eq("ticker_symbol")
    is_contraction_tail = d["matched_text_norm"].isin(CONTRACTION_TAILS)
    drop_mask = (
        is_ticker
        & is_symbol
        & is_contraction_tail
        & (d["entity_id"].astype(str).str.len() == 1)
    )
    return d.loc[~drop_mask].copy()


GROUP_COLS = ["ts", "user_id", "chat_id", "trace_id"]


def build_group_payloads(
    raw_df: pd.DataFrame, max_candidates_per_msg: int = 30
) -> pd.DataFrame:
    rows = []
    for key, g in raw_df.groupby(GROUP_COLS, sort=False):
        message = str(g["msg"].iloc[0])
        g_best = (
            g.sort_values("confidence", ascending=False)
            .drop_duplicates(
                subset=["entity_type", "entity_id", "matched_text_norm"],
                keep="first",
            )
            .sort_values("confidence", ascending=False)
            .head(max_candidates_per_msg)
        )

        payload = []
        for idx, r in enumerate(g_best.itertuples(index=False)):
            payload.append(
                {
                    "idx": idx,
                    "entity_type": r.entity_type,
                    "entity_id": str(r.entity_id),
                    "canonical": str(r.canonical),
                    "matched_text_norm": str(r.matched_text_norm),
                    "match_source": str(r.match_source),
                    "confidence": float(r.confidence),
                    "context": str(r.context),
                }
            )

        if not payload:
            continue

        rows.append(
            {
                "ts": key[0],
                "user_id": key[1],
                "chat_id": key[2],
                "trace_id": key[3],
                "msg": message,
                "payload": payload,
            }
        )
    return pd.DataFrame(rows)
