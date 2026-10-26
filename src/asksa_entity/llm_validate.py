from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# LLM_SYSTEM_BALANCED = """You validate extracted entities from a FINANCIAL MARKETS Q&A chat.

# DOMAIN CONTEXT
# - Messages are usually about public companies, stocks/ETFs, sectors, and market themes.
# - Users often write short, informal requests (e.g., "AAPL?", "Apple analysis", "Thoughts on Tesla").
# - Users also write comparisons with mixed casing: "kalu vs AA", "Strl vs Dy vs mrz".

# GOAL
# Balance precision and recall:
# - Keep an entity when it is reasonably likely the user meant that entity.
# - Reject only when there is clear evidence it is NOT that entity.

# INPUT
# You receive:
# 1) FULL message text
# 2) A list of extracted candidates with fields: (idx, entity_type, entity_id, canonical, matched_text, source, context)

# TASK
# Return only the candidates that are intended in the message context.

# OUTPUT (STRICT JSON)
# Return ONLY JSON, no extra text.
# Schema:
# {"entities": [{"idx": 0, "entity_type": "ticker", "entity_id": "AAPL"}]}
# If none are valid: {"entities": []}

# -------------------------
# CORE DECISION HEURISTICS
# -------------------------
# KEEP if at least one of these is true:
# - The candidate matches the user's apparent intent: analysis, outlook, news, price, buy/sell, valuation, earnings, etc.
# - The candidate appears as a clean standalone mention (ticker token or full company/topic phrase), even if no finance keywords exist.

# REJECT if any of these is true:
# - The match is clearly a grammatical word/auxiliary/conjunction/determiner usage.
# - The match is clearly a contraction artifact (see below).
# - The match is embedded inside another word, or is part of a person name / non-financial named entity.

# IMPORTANT: Sentence-start capitalization is NOT evidence of a ticker.
# If a token is TitleCase only because it is the first word, treat it as normal text unless other ticker signals exist.

# -------------------------
# TICKER RULES
# -------------------------
# A ticker candidate is VALID if it likely refers to a listed instrument in this message.

# STRONG KEEP signals:
# - The token appears as a standalone ticker-like token (AAPL, TSLA, BRK.B, RDS.A, BTC, ETH, QNT-USD, etc.)
# - It appears in a list of tickers, or in "COMPANY (TICKER)" patterns
# - Nearby finance cues: stock, shares, ticker, price, chart, earnings, options, valuation, buy/sell, etc.
# - A '$' prefix is a strong signal: $AAPL, $TSLA

# MEDIUM KEEP signals (allowed):
# - The message is clearly asking for analysis/outlook and the token is a plausible ticker/company mention
#   Examples: "AAPL analysis", "Thoughts on TSLA", "nvda outlook"

# HARD REJECT (always reject):
# - Contraction tails from apostrophes: I'm / it's / we're / they've / you'll -> 'm, 's, 're, 've, 'd, 'll
#   These are NEVER tickers.

# -------------------------
# RATING / SCREENER TERMS (NEW)
# -------------------------
# Some capitalized finance words are NOT tickers in common user phrasing.

# HARD REJECT as ticker if matched_text/canonical is "Quant" or "quant" AND the message suggests ratings/screeners, e.g. contains any of:
# - "strong buy", "strong buys", "strong return", "quant rating", "quant grade", "quant score", "factor grade"
# - "value", "growth", "profitability", "momentum", "revisions" (as rating categories)
# In those cases, "Quant" refers to a rating/screener concept, NOT QNT-USD.
# (Only keep QNT-USD if the message explicitly indicates crypto/ticker usage, e.g. "QNT", "QNT-USD", "$QNT", "Quant (QNT)", or obvious crypto wording.)

# Example:
# - "I want Quant strong buys" -> REJECT ticker candidate "Quant"

# -------------------------
# COMPARISON PATTERNS (NEW)
# -------------------------
# If the message contains comparison/list patterns like:
# - "X vs Y", "X vs Y vs Z", "X v Y", "X versus Y"
# then tokens around these separators are likely tickers, EVEN if they are lowercase or mixed case.
# KEEP such candidates when they are standalone tokens (surrounded by spaces/punctuation) and look ticker-like.

# Examples:
# - "kalu vs AA" -> KEEP KALU and AA
# - "Strl vs Dy vs mrz" -> KEEP STRL, DY, MRZ (but do NOT invent tickers; only validate candidates)

# -------------------------
# AMBIGUOUS SHORT TICKERS (1-2 letters)
# -------------------------
# Single-letter or 2-letter tickers are ambiguous (M, F, T, IT, ON, AS, …).

# Default rule:
# - KEEP only if at least one reasonable signal exists:
#   - standalone token (exact) AND (finance intent words OR in a ticker list OR in parentheses OR '$' prefix OR in a VS-comparison pattern)
# - Otherwise REJECT.

# SHORT WORD STOPLIST (NEW):
# Always REJECT as ticker if the token is a common short word/pronoun/preposition and there are no explicit ticker signals.
# Examples (case-insensitive): me, mi, my, we, us, it, as, at, in, on, or, to, of, an, a
# (If such a token is clearly used as a ticker with explicit signals like "$IT" or "IT (ticker)", you may KEEP.)

# This specifically prevents false positives like "mi" being treated as a ticker.

# -------------------------
# BUNDLES (entity_id contains '|')
# -------------------------
# - If entity_id contains '|', choose the ONE most likely ticker based on the message text.
# - If there is no way to choose, keep the bundle ONLY if the message clearly refers to the company name phrase and ambiguity is acceptable;
#   otherwise reject.

# NEVER-TICKER TOKENS (HARD REJECT)
# Always REJECT as ticker if matched_text/canonical equals (case-insensitive):
# - "vs", "v", "versus"
# These are comparison separators in user queries, not intended tickers in this product.

# RATINGS / SCREENER MODE (HARD RULES)
# If the message is about ratings/screeners (mentions any of: rating/ratings, strong buy, quant, score, grade, momentum, factor, revisions, profitability, value, growth):
# - Treat "quant", "sa", "wall", "wall st", "rating", "score", "grade", "momentum" as NON-TICKER terms.
# - Only KEEP tickers that are explicit instrument mentions (standalone ticker-like tokens, $TICKER, (TICKER), or clearly a company name).
# - Specifically: if matched_text contains "quant" or "rating" or "score" or "grade" (even as abbreviations like "r."), REJECT it as ticker.

# HARD REJECT as ticker if matched_text/canonical contains "quant" in any form:
# "quant", "quant.", "quant r", "quant r.", "quant rating", "quant score", "quant grade"
# UNLESS the message explicitly includes QNT, QNT-USD, $QNT, or "Quant (QNT)".

# -------------------------
# SECTOR / INDUSTRY / TOPIC RULES
# -------------------------
# - Keep if the phrase is used as a category/theme in the message (explicitly or implicitly).
#   Examples: "energy sector", "semiconductor stocks", "AI theme", "defense companies", "cloud providers"
# - If the phrase is used as a generic adjective/noun with no category intent, reject.
# """

LLM_SYSTEM_BALANCED = """You validate extracted entities from a financial markets chat.

INPUT:
(1) full message text
(2) candidates: (idx, entity_type, entity_id, canonical, matched_text, source, context)

TASK:
Select only candidates the user intended. Do NOT invent entities. Use only given candidates.

OUTPUT: STRICT JSON ONLY
{"entities":[{"idx":0,"entity_type":"ticker","entity_id":"AAPL"}]}
If none: {"entities":[]}

GENERAL
- KEEP if the candidate is reasonably likely intended in this message.
- REJECT if it is clearly a normal English word usage, a contraction artifact, embedded in another word, or a non-financial named entity/person.

TICKERS
KEEP ticker if at least one strong signal:
- standalone ticker-like token (letters/digits with optional . or -), e.g. AAPL, BRK.B, RDS.A, QNT-USD
- appears in a list or comparison pattern (X vs Y vs Z / v / versus)
- has $ prefix or (TICKER) pattern
- nearby finance intent words: stock, shares, price, chart, earnings, options, valuation, buy/sell, news, outlook, analysis

NOTES:
- Sentence-start TitleCase alone is NOT a ticker signal.
- NEVER keep "vs", "v", "versus" as tickers.

CONTRACTIONS (ALWAYS REJECT as ticker)
'm, 's, 're, 've, 'd, 'll

AMBIGUOUS SHORT TICKERS (1–2 chars)
KEEP only with explicit ticker signals (standalone + one of: finance intent, list/comparison, parentheses, $ prefix). Otherwise REJECT.
Also REJECT common short words unless explicit signals: me, mi, my, we, us, it, as, at, in, on, or, to, of, an, a.

BUNDLES (entity_id contains '|')
Choose the ONE best matching the message text if possible. If not decidable, keep the bundle only when the message clearly refers to the company phrase; else reject.

RATINGS/SCREENER MODE
If the message mentions any of: rating/ratings, strong buy, quant, score, grade, factor, momentum, revisions, profitability, value, growth
- Treat "quant", "rating", "score", "grade", "momentum" as NON-ticker concepts.
- Reject any candidate whose matched_text/canonical contains "quant"
  UNLESS message explicitly indicates crypto ticker QNT: "QNT", "QNT-USD", "$QNT", or "Quant (QNT)".

SECTOR/INDUSTRY/TOPIC
Keep if used as a category/theme (sector/theme/stocks/companies/etc.). Reject if used as generic adjective with no category intent.

"""


_FULL_OUTPUT_RULES = """OUTPUT: STRICT JSON ONLY
{"entities":[{"idx":0,"entity_type":"ticker","entity_id":"AAPL"}]}
If none: {"entities":[]}"""
_COMPACT_OUTPUT_RULES = """OUTPUT: STRICT JSON ONLY, the idx of every candidate to keep (nothing else)
{"keep":[0,2]}
If none: {"keep":[]}"""
_FULL_BUNDLE_RULES = """Choose the ONE best matching the message text if possible. If not decidable, keep the bundle only when the message clearly refers to the company phrase; else reject."""
_COMPACT_BUNDLE_RULES = """Choose the ONE best matching the message text if possible and write it as "idx:TICKER", e.g. {"keep":[0,"3:BRK.B"]}. If not decidable, keep the bundle (just its idx) only when the message clearly refers to the company phrase; else reject."""

assert (
    _FULL_OUTPUT_RULES in LLM_SYSTEM_BALANCED
    and _FULL_BUNDLE_RULES in LLM_SYSTEM_BALANCED
)

# Same rules as LLM_SYSTEM_BALANCED, only the answer format differs: the model
# returns candidate numbers (~2 tokens each) instead of full objects (~23
# tokens each), and cannot copy candidate fields or misspell ids.
LLM_SYSTEM_COMPACT = LLM_SYSTEM_BALANCED.replace(
    _FULL_OUTPUT_RULES, _COMPACT_OUTPUT_RULES
).replace(_FULL_BUNDLE_RULES, _COMPACT_BUNDLE_RULES)

SYSTEM_PROMPTS = {"full": LLM_SYSTEM_BALANCED, "compact": LLM_SYSTEM_COMPACT}
# The model's answer is started with this text, so it can only continue the JSON
ANSWER_PREFIXES = {"full": '{"entities":[', "compact": '{"keep":['}


def build_prompts(
    messages: Sequence[str],
    payloads: Sequence[List[Dict[str, Any]]],
    output_format: str = "full",
) -> List[str]:
    """LLM prompts for messages and their candidates in the given answer format."""
    system_prompt = SYSTEM_PROMPTS[output_format]
    return [
        build_llm_prompt(m, p, system_prompt=system_prompt)
        for m, p in zip(messages, payloads)
    ]


def build_llm_prompt(
    message: str,
    candidates_payload: List[Dict[str, Any]],
    system_prompt: str = LLM_SYSTEM_BALANCED,
) -> str:
    lines = []
    for c in candidates_payload:
        eid = str(c["entity_id"])
        et = str(c["entity_type"])
        matched = str(c.get("matched_text_norm", ""))
        src = str(c.get("match_source", ""))
        conf = float(c.get("confidence", 0.0))

        hints = []
        if et == "ticker":
            if len(eid) == 1:
                hints.append("single_letter_ticker")
            elif len(eid) <= 2:
                hints.append("short_ticker")
            if "|" in eid:
                hints.append("bundle")
            if matched in {"m", "s", "re", "ve", "d", "ll"}:
                hints.append("contraction_like")

        lines.append(
            f"- idx={c['idx']} type={et} id={eid} canonical={c.get('canonical', '')!r} "
            f"matched={matched!r} source={src} conf={conf:.2f} hints={','.join(hints)} "
            f"context={c.get('context', '')!r}"
        )

    return f"""{system_prompt}

Message:
{message}

Candidates:
{chr(10).join(lines)}

Return ONLY JSON:
"""


_JSON_FENCE_RE = re.compile(r"```json\s*([\s\S]*?)\s*```", flags=re.IGNORECASE)


def unwrap_singleton_lists(x: Any) -> Any:
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def get_generated_text(x: Any) -> str:
    x = unwrap_singleton_lists(x)
    if isinstance(x, dict) and "generated_text" in x:
        return x["generated_text"]
    if (
        isinstance(x, list)
        and x
        and isinstance(x[0], dict)
        and "generated_text" in x[0]
    ):
        return x[0]["generated_text"]
    return str(x)


def extract_first_json_object(text: str) -> Dict:
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found (no '{').")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    return obj


def parse_llm_entities_from_text(text: str) -> List[Dict[str, str]]:
    obj = extract_first_json_object(text)
    ents = obj.get("entities") or []
    out = []
    for e in ents:
        try:
            idx = int(e.get("idx"))
        except Exception:
            continue
        et = str(e.get("entity_type", "")).strip()
        eid = str(e.get("entity_id", "")).strip()
        if et and eid:
            out.append({"idx": idx, "entity_type": et, "entity_id": eid})
    return out


# A complete entity object; extra fields after entity_id are allowed (the
# model sometimes copies the candidate's canonical/matched_text/... too)
_ENTITY_OBJ_RE = re.compile(
    r'\{\s*"idx"\s*:\s*(\d+)\s*,\s*"entity_type"\s*:\s*"([^"]+)"\s*,\s*"entity_id"\s*:\s*"([^"]+)"[^{}]*\}'
)
# A complete item of a compact "keep" list: 3 or "3" or "3:BRK.B", followed by , or ]
_KEEP_ITEM_RE = re.compile(
    r'(?:"\s*(\d+)\s*(?::\s*([^"]*?))?\s*"|(\d+))\s*(?=[,\]])'
)


def _keep_item(value: Any) -> Optional[Dict[str, Any]]:
    """Compact item -> entity with idx and optional bundle choice (type/id come from the candidate)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return {"idx": value, "entity_type": None, "entity_id": None}
    m = re.fullmatch(r"\s*(\d+)\s*(?::\s*(.*?))?\s*", str(value))
    if not m:
        return None
    return {
        "idx": int(m.group(1)),
        "entity_type": None,
        "entity_id": (m.group(2) or None),
    }


def parse_keep_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse a compact answer {"keep":[0,"3:BRK.B"]}."""
    obj = extract_first_json_object(text)
    keep = obj.get("keep") or []
    return [e for e in (_keep_item(v) for v in keep) if e is not None]


def salvage_entities_from_text(
    text: str, output_format: str = "full"
) -> List[Dict[str, Any]]:
    """Recover complete entities from an answer that was cut off.

    Only items that were fully written are returned; a half-written last one
    is ignored.
    """
    if output_format == "compact":
        return [
            {
                "idx": int(q_idx or n_idx),
                "entity_type": None,
                "entity_id": (choice or None),
            }
            for q_idx, choice, n_idx in _KEEP_ITEM_RE.findall(text)
        ]
    return [
        {"idx": int(idx), "entity_type": et.strip(), "entity_id": eid.strip()}
        for idx, et, eid in _ENTITY_OBJ_RE.findall(text)
    ]


def answer_stop_reason(
    text: str, max_entities: Optional[int] = None, output_format: str = "full"
) -> Optional[str]:
    """Why an answer can stop now, or None to keep generating.

    "json_closed": the first JSON object is complete; the parser ignores
        anything after it, so further tokens only cost time.
    "repeating": an entity idx was written twice, or more entities than
        there are candidates; a correct answer never does either, so the
        model is looping (complete items so far are kept by salvage).
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escaped = False
    for ch in text[start:]:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "json_closed"

    idxs = [e["idx"] for e in salvage_entities_from_text(text, output_format)]
    if len(idxs) != len(set(idxs)) or (
        max_entities is not None and len(idxs) > max_entities
    ):
        return "repeating"
    return None


class _AnswerStoppingCriteria:
    """Per-row stopping for generate(): stops a row as soon as answer_stop_reason() fires.

    The text is decoded only when the newest token contains a character that
    can complete an item or the JSON ("}", "]" or ","), so the check is cheap.
    """

    TRIGGER_CHARS = ("}", "]", ",")

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        max_entities: List[Optional[int]],
        prefix: str,
        output_format: str,
    ):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_entities = max_entities
        self.prefix = prefix
        self.output_format = output_format
        self.reasons: List[Optional[str]] = [None] * len(max_entities)

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        done = []
        last_ids = input_ids[
            :, -1
        ].tolist()  # one GPU sync per step, not per row
        for r, last_id in enumerate(last_ids):
            if (
                self.reasons[r] is None
                and input_ids.shape[1] > self.prompt_len
            ):
                last = str(self.tokenizer.convert_ids_to_tokens(last_id))
                if any(ch in last for ch in self.TRIGGER_CHARS):
                    text = self.prefix + self.tokenizer.decode(
                        input_ids[r, self.prompt_len :],
                        skip_special_tokens=True,
                    )
                    self.reasons[r] = answer_stop_reason(
                        text, self.max_entities[r], self.output_format
                    )
            done.append(self.reasons[r] is not None)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def load_llm_pipeline(model_name: str, token: str):
    """
    Uses your preferred loading style (AutoTokenizer + AutoModelForCausalLM, device_map="auto").
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=os.getenv("HF_TOKEN"),
    )
    model.eval()

    return pipeline("text-generation", model=model, tokenizer=tok)


# def load_llm_pipeline(model_name: str):
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

#     tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     tok.padding_side = "left"
#     if tok.pad_token_id is None:
#         tok.pad_token = tok.eos_token

#     # 4-bit quant config
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",          # best default
#         bnb_4bit_use_double_quant=True,     # better accuracy
#         bnb_4bit_compute_dtype=torch.float16
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         quantization_config=bnb_config,
#         dtype=torch.float16,
#         low_cpu_mem_usage=True
#     )
#     model.eval()

#     return pipeline("text-generation", model=model, tokenizer=tok)


def run_llm_batched(
    llm_pipe,
    groups_df: pd.DataFrame,
    batch_size: int = 4,
    max_new_tokens: int = 120,
    repetition_penalty: float = 1.05,
) -> List[Dict[str, Any]]:
    prompts = groups_df["prompt"].tolist()

    gen_kwargs = dict(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
    )

    try:
        raw_out = llm_pipe(prompts, **gen_kwargs, return_full_text=False)
    except TypeError:
        raw_out = llm_pipe(prompts, **gen_kwargs)

    cleaned = []
    for i, item in enumerate(raw_out):
        raw_text = get_generated_text(item)

        # Remove prompt if it leaked into output
        prompt = prompts[i]
        if raw_text.startswith(prompt):
            raw_text = raw_text[len(prompt) :].lstrip()

        # Parse FIRST JSON only; ignore any extra "thinking"/explanations
        try:
            ents = parse_llm_entities_from_text(raw_text)
            cleaned.append(
                {"text": raw_text, "entities": ents, "parse_ok": True}
            )
        except Exception:
            cleaned.append(
                {"text": raw_text, "entities": [], "parse_ok": False}
            )

    return cleaned


OUTPUT_ENTITY_COLUMNS = [
    "ts",
    "user_id",
    "chat_id",
    "trace_id",
    "msg",
    "entity_type",
    "entity_id",
]


def resolve_entity(
    item: Dict[str, Any], payload: List[Dict[str, Any]]
) -> Tuple[str, str, bool]:
    """Map one LLM-selected entity to a real candidate: (entity_type, entity_id, exact).

    1) The LLM names a candidate exactly (type + id, or one member of a
       bundle "A|B") -> that candidate. Same result as the original code.
    2) Otherwise (misspelled / invented id or type) -> the candidate at idx,
       so only ids that really came from matching ever reach the table.
    exact=False means case 2 was used.
    """
    if item.get("entity_type") is None:
        # compact answer: type/id from the candidate, optional bundle choice
        c = payload[item["idx"]]
        ctype, cid = str(c["entity_type"]), str(c["entity_id"])
        choice = str(item.get("entity_id") or "").strip().upper()
        if "|" in cid and choice:
            for part in cid.split("|"):
                if part.upper() == choice:
                    return ctype, part, True
        return ctype, cid, True

    llm_type = str(item.get("entity_type", "")).strip()
    llm_id = str(item.get("entity_id", "")).strip()

    for c in payload:
        ctype, cid = str(c["entity_type"]), str(c["entity_id"])
        if ctype != llm_type:
            continue
        if cid == llm_id:
            return ctype, cid, True
        if "|" in cid:
            for part in cid.split("|"):
                if part.upper() == llm_id.upper():
                    return ctype, part, True

    c = payload[item["idx"]]
    return str(c["entity_type"]), str(c["entity_id"]), False


def build_validated_df(
    groups_df: pd.DataFrame,
    llm_outputs: List[Dict[str, Any]],
    debug: bool = False,
) -> pd.DataFrame:
    """One row per (message, accepted entity); always has OUTPUT_ENTITY_COLUMNS."""
    assert len(groups_df) == len(llm_outputs)
    rows = []
    out_of_range = 0
    corrected = 0
    bundles_narrowed = 0
    for g, out in zip(groups_df.itertuples(index=False), llm_outputs):
        payload = g.payload
        # tickers the user wrote themselves (plain, not a bundle)
        plain_tickers = {
            str(c["entity_id"])
            for c in payload
            if c["entity_type"] == "ticker" and "|" not in str(c["entity_id"])
        }
        for item in out["entities"]:
            idx = item["idx"]
            if not (0 <= idx < len(payload)):
                out_of_range += 1
                if debug:
                    print(
                        "idx out of range:", idx, "payload_len:", len(payload)
                    )
                continue
            etype, eid, exact = resolve_entity(item, payload)
            corrected += not exact
            # A kept bundle (e.g. found by company name "Adobe Inc." ->
            # "ADBE|ADBE:CA|ZADB:CA") whose member is also a plain candidate
            # of this message ("ADBE") means that member.
            if etype == "ticker" and "|" in eid:
                members = [m for m in eid.split("|") if m in plain_tickers]
                if members:
                    eid = members[0]
                    bundles_narrowed += 1
            rows.append(
                {
                    "ts": g.ts,
                    "user_id": g.user_id,
                    "chat_id": g.chat_id,
                    "trace_id": g.trace_id,
                    "msg": g.msg,
                    "entity_type": etype,
                    "entity_id": eid,
                }
            )

    if out_of_range or corrected or bundles_narrowed:
        logger.info(
            f"LLM entities with idx out of range (dropped): {out_of_range}, "
            f"not matching any candidate (replaced by candidate at idx): {corrected}, "
            f"bundles narrowed to a ticker written in the message: {bundles_narrowed}"
        )
    df = pd.DataFrame(rows, columns=OUTPUT_ENTITY_COLUMNS)
    return df.drop_duplicates(subset=OUTPUT_ENTITY_COLUMNS).reset_index(
        drop=True
    )


class EntityValidatorLLM:
    """Validates candidate entities with an instruction-tuned LLM (direct generate, no pipeline).

    use_chat_template=True: the prompt goes into the model's chat template
        (the format Gemma-it was trained on) and generation stops at
        <end_of_turn>, so max_new_tokens can be large without wasting time.
    use_chat_template=False: reproduces the original notebook (raw prompt as
        a plain text completion via the text-generation pipeline).

    Rows are sorted by prompt length and batched so that
    rows_in_batch * longest_prompt <= max_batch_tokens (None = fixed batches
    in input order). Prompts longer than max_input_tokens are skipped
    (reported as skipped=True), never truncated.

    Answer length: with new_tokens_per_candidate set and n_candidates passed
    to run(), a row may generate at most
    new_tokens_base + new_tokens_per_candidate * n_candidates tokens (capped
    by max_new_tokens): a correct answer never lists more entities than there
    are candidates, so a looping answer stops early instead of holding the
    whole batch for max_new_tokens steps.

    salvage_truncated_json: if the JSON is cut off, keep the entity objects
    that were written completely (reported as salvaged=True).

    stop_on_complete_json: stop each row as soon as its JSON object is
    complete or it starts repeating entities (see answer_stop_reason); the
    parsed result is the same, only wasted tokens are skipped.

    output_format: "full" ({"entities":[{idx, entity_type, entity_id}]},
    the original) or "compact" ({"keep":[idx, "idx:TICKER"]}); prompts must
    be built for the same format (build_prompts).
    prefill_answer: start the model's answer with the JSON opening
    (ANSWER_PREFIXES). Off by default: on real data it made the model keep
    candidates in almost every message (after an open "[" it rarely closes
    the list empty), i.e. it biased the decisions.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-9b-it",
        token: Optional[str] = None,
        use_chat_template: bool = True,
        max_new_tokens: int = 1280,
        repetition_penalty: float = 1.0,
        batch_size: int = 8,
        max_batch_tokens: Optional[int] = 5120,
        max_input_tokens: Optional[int] = 5120,
        new_tokens_base: int = 32,
        new_tokens_per_candidate: Optional[int] = 40,
        salvage_truncated_json: bool = True,
        stop_on_complete_json: bool = True,
        output_format: str = "full",
        prefill_answer: bool = False,
        dtype: Any = None,
        allow_cpu_offload: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.use_chat_template = use_chat_template
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_input_tokens = max_input_tokens
        self.new_tokens_base = new_tokens_base
        self.new_tokens_per_candidate = new_tokens_per_candidate
        self.salvage_truncated_json = salvage_truncated_json
        self.stop_on_complete_json = stop_on_complete_json
        if output_format not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown output_format: {output_format}")
        if output_format == "compact" and not use_chat_template:
            raise ValueError(
                "compact output_format is only supported with the chat template"
            )
        self.output_format = output_format
        self.answer_prefix = (
            ANSWER_PREFIXES[output_format]
            if (prefill_answer and use_chat_template)
            else ""
        )

        if dtype is None:
            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            else:
                dtype = torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        free_gb = None
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            logger.info(f"Free GPU memory before loading: {free_gb:.1f} GB")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=dtype,
            token=token,
        )
        self.model.eval()

        device_map = getattr(self.model, "hf_device_map", None) or {}
        offloaded = [
            name for name, dev in device_map.items() if dev in ("cpu", "disk")
        ]
        if offloaded and not allow_cpu_offload:
            raise RuntimeError(
                f"Model does not fit on the GPU: {len(offloaded)} of {len(device_map)} modules "
                f"were offloaded to CPU/disk (free GPU memory before loading: {free_gb}). "
                "Another model is probably still in GPU memory: restart the kernel / process."
            )

        self.stop_token_ids = self._stop_token_ids()
        logger.info(
            f"Loaded LLM: {model_id} ({dtype}), chat template: {use_chat_template}"
        )

    def _stop_token_ids(self) -> List[int]:
        ids = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if (
            isinstance(eot, int)
            and eot != self.tokenizer.unk_token_id
            and eot not in ids
        ):
            ids.append(eot)
        return ids

    # ---- prompt rendering / tokenization ----

    def _render(self, prompts: Sequence[str]) -> List[str]:
        if not self.use_chat_template:
            return list(prompts)
        # Gemma has no system role: the whole prompt goes into the user turn
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.answer_prefix
            for p in prompts
        ]

    def _tokenize(self, rendered: Sequence[str], **kwargs):
        # The chat template already starts with <bos>; a raw prompt needs the tokenizer to add it
        return self.tokenizer(
            list(rendered),
            add_special_tokens=not self.use_chat_template,
            **kwargs,
        )

    def prompt_lengths(self, prompts: Sequence[str]) -> List[int]:
        """Token length of each prompt as the model gets it."""
        return [
            len(ids)
            for ids in self._tokenize(self._render(prompts))["input_ids"]
        ]

    # ---- batching / generation ----

    def _make_batches(self, lengths: Dict[int, int]) -> List[List[int]]:
        """Group row positions (keys of `lengths`) into batches."""
        positions = list(lengths)
        if not self.max_batch_tokens:
            return [
                positions[s : s + self.batch_size]
                for s in range(0, len(positions), self.batch_size)
            ]

        # longest first: if memory is too tight, it fails in the first batch
        order = sorted(positions, key=lambda i: lengths[i], reverse=True)
        batches: List[List[int]] = []
        current: List[int] = []
        longest = 0
        for i in order:
            new_longest = max(longest, lengths[i])
            too_many = len(current) >= self.batch_size
            too_long = (len(current) + 1) * new_longest > self.max_batch_tokens
            if current and (too_many or too_long):
                batches.append(current)
                current = []
                new_longest = lengths[i]
            current.append(i)
            longest = new_longest
        if current:
            batches.append(current)
        return batches

    def _generate(
        self,
        rendered: List[str],
        max_new_tokens: int,
        max_entities: List[Optional[int]],
    ) -> Tuple[List[str], List[str]]:
        """Generate for one batch: (texts, stop_reasons).

        stop_reason: "eos" (model ended its turn), "json_closed",
        "repeating" (see answer_stop_reason) or "limit" (max_new_tokens hit).
        """
        import torch
        from transformers import StoppingCriteriaList

        enc = self._tokenize(rendered, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        extra = {}
        criteria = None
        if self.stop_on_complete_json:
            criteria = _AnswerStoppingCriteria(
                self.tokenizer,
                prompt_len,
                max_entities,
                self.answer_prefix,
                self.output_format,
            )
            extra["stopping_criteria"] = StoppingCriteriaList([criteria])

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.stop_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                **extra,
            )
        gen = out[:, prompt_len:]
        texts = [
            self.answer_prefix + t
            for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        ]
        stop = torch.tensor(self.stop_token_ids, device=gen.device)
        has_eos = torch.isin(gen, stop).any(dim=1).tolist()

        reasons = []
        for r, eos in enumerate(has_eos):
            if criteria is not None and criteria.reasons[r] is not None:
                reasons.append(criteria.reasons[r])
            else:
                reasons.append("eos" if eos else "limit")
        return texts, reasons

    def _row_limit(self, n_candidates: Optional[int]) -> int:
        if n_candidates is None or not self.new_tokens_per_candidate:
            return self.max_new_tokens
        return min(
            self.max_new_tokens,
            self.new_tokens_base
            + self.new_tokens_per_candidate * n_candidates,
        )

    def _parse(self, text: str) -> Tuple[List[Dict[str, Any]], bool, bool]:
        """(entities, parse_ok, salvaged)."""
        try:
            if self.output_format == "compact":
                return parse_keep_from_text(text), True, False
            return parse_llm_entities_from_text(text), True, False
        except Exception:
            if self.salvage_truncated_json:
                ents = salvage_entities_from_text(text, self.output_format)
                if ents:
                    return ents, False, True
            return [], False, False

    def run(
        self,
        prompts: Sequence[str],
        n_candidates: Optional[Sequence[int]] = None,
        desc: str = "Validating entities",
    ) -> List[Dict[str, Any]]:
        """Validate all prompts; results are in input order.

        n_candidates: candidates per prompt, bounds each answer's length.
        Each result: text, entities, parse_ok, salvaged, skipped, finished,
        stop_reason, prompt_tokens, max_new_tokens. finished=False means the
        answer was cut (token limit or looping).
        """
        rendered = self._render(prompts)
        lengths = [len(ids) for ids in self._tokenize(rendered)["input_ids"]]
        limits = [
            self._row_limit(None if n_candidates is None else n_candidates[i])
            for i in range(len(rendered))
        ]

        results: List[Optional[Dict[str, Any]]] = [None] * len(rendered)
        eligible: Dict[int, int] = {}
        for i, n_tokens in enumerate(lengths):
            if self.max_input_tokens and n_tokens > self.max_input_tokens:
                results[i] = {
                    "text": "",
                    "entities": [],
                    "parse_ok": False,
                    "salvaged": False,
                    "skipped": True,
                    "finished": True,
                    "stop_reason": "skipped",
                    "prompt_tokens": n_tokens,
                    "max_new_tokens": limits[i],
                }
            else:
                eligible[i] = n_tokens

        for batch in tqdm(
            self._make_batches(eligible), desc=desc, unit="batch"
        ):
            batch_limit = max(limits[i] for i in batch)
            max_entities = [
                None if n_candidates is None else n_candidates[i]
                for i in batch
            ]
            texts, reasons = self._generate(
                [rendered[i] for i in batch], batch_limit, max_entities
            )
            for i, text, reason in zip(batch, texts, reasons):
                ents, ok, salvaged = self._parse(text)
                results[i] = {
                    "text": text,
                    "entities": ents,
                    "parse_ok": ok,
                    "salvaged": salvaged,
                    "skipped": False,
                    "finished": reason in ("eos", "json_closed"),
                    "stop_reason": reason,
                    "prompt_tokens": lengths[i],
                    "max_new_tokens": limits[i],
                }
        return results
