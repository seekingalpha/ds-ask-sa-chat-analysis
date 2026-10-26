"""NER job: extract entities from first chat messages and load to Redshift.

Production version of notebooks/entity_extraction.ipynb:
dictionary/fuzzy matching builds candidates, the LLM keeps the intended ones.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from asksa_entity import (
    EntityValidatorLLM,
    ExtractConfig,
    build_group_payloads,
    build_prompts,
    build_resources,
    build_validated_df,
    drop_contraction_artifacts,
    extract_raw_matches,
    get_topics_df,
)
from asksa_jobs.common import (
    Settings,
    export_hf_token,
    fetch_df,
    load_df_to_redshift,
    read_query,
)

logger = logging.getLogger(__name__)

MESSAGES_QUERY_FILE = "get_all_user_chats.sql"
TICKERS_QUERY_FILE = "get_tickers.sql"
S3_FILE_NAME = "result_er.csv"
MESSAGE_KEY = ["ts", "user_id", "chat_id", "trace_id"]

# Same order as ds.production__ask_sa_message_entity_extraction
OUTPUT_COLUMNS = [
    "ts",
    "user_id",
    "chat_id",
    "trace_id",
    "msg",
    "entity_type",
    "entity_id",
    "msg_date",
]
# msg is varchar(MAX) = 65535 bytes; 16000 chars stay below it even at
# 4 bytes per char
MSG_MAX_CHARS = 16000
ENTITY_ID_MAX_CHARS = 100  # entity_id varchar(100)
ENTITY_TYPE_MAX_CHARS = 12  # entity_type varchar(12)


@dataclass(frozen=True)
class NerParams:
    """Matching and LLM settings for the NER job.

    Matching settings and validation rules are copied from the notebook,
    plus ticker_noise_filter: ordinary words matched as tickers ("news" ->
    NE.WS, "hold" -> HOLD, "or" -> OR) are dropped before the LLM when
    nothing suggests a ticker (see asksa_entity/ticker_noise.py).
    LLM settings differ:
    output_format: "compact" = the model answers with candidate numbers
        {"keep":[0,"3:BRK.B"]} (~2 tokens each) instead of full objects
        (~23 tokens each, the notebook's "full" format). Tested on the same
        sample: ~6x faster, and "full" keeps only the first few tickers of
        long pasted ticker lists (the model stops writing the long JSON),
        while "compact" keeps the whole list. Same validation rules.
    max_new_tokens: the notebook used 30, which cut every answer with 2+
        entities (~29 tokens for 1 entity, ~52 for 2, measured). Now each
        answer may take new_tokens_base + new_tokens_per_candidate *
        n_candidates tokens (capped at max_new_tokens), and stops as soon as
        its JSON is complete. For "full" use 32 + 40 per candidate.
    prefill_answer: off; starting the answer with the JSON opening made the
        model keep candidates in almost every message (tested).
    repetition_penalty: 1.0 instead of 1.05; the JSON repeats the same keys
        for every entity, and the penalty pushes the model away from them.
    max_batch_tokens / max_input_tokens: same memory envelope as the
        sentiment job (verified on L4); the longest prompt seen was ~5050
        tokens.
    """

    model_id: str = "google/gemma-2-9b-it"
    # matching
    extract_batch_size: int = 5000
    max_ngram_fallback: int = 6
    max_candidates_per_msg: int = 30
    ticker_noise_filter: bool = True
    # LLM
    use_chat_template: bool = True
    output_format: str = "compact"
    prefill_answer: bool = False
    max_new_tokens: int = 1280
    new_tokens_base: int = 16
    new_tokens_per_candidate: int = 12
    repetition_penalty: float = 1.0
    batch_size: int = 8
    max_batch_tokens: int = 5120
    max_input_tokens: int = 5120


def require_gpu() -> None:
    """Fail fast instead of running a 9B model on CPU for hours."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, refusing to run on CPU")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def load_messages(settings: Settings) -> pd.DataFrame:
    """Fetch first messages of new chats, one row per message.

    Sorted, because the query has no ORDER BY: runs (and --max-rows
    samples) are reproducible.
    """
    messages = fetch_df(settings, read_query(settings, MESSAGES_QUERY_FILE))
    before = len(messages)
    messages = messages.drop_duplicates(subset=MESSAGE_KEY + ["msg"])
    messages = messages.sort_values(MESSAGE_KEY, kind="stable")
    logger.info(
        f"New messages: {len(messages)} (duplicates dropped: {before - len(messages)})"
    )
    return messages.reset_index(drop=True)


def build_candidates(
    settings: Settings, messages: pd.DataFrame, params: NerParams
) -> pd.DataFrame:
    """Dictionary/fuzzy matching: one row per message with candidates and prompt."""
    tickers = fetch_df(settings, read_query(settings, TICKERS_QUERY_FILE))
    tickers_df = tickers[["ticker_id", "company_name"]].copy()
    secind_df = tickers[["sector_name", "industry_name"]].drop_duplicates()
    topics_df = get_topics_df()
    logger.info(
        f"Dictionaries: {len(tickers_df)} tickers, {len(secind_df)} sector/industry "
        f"rows, {len(topics_df)} topics"
    )

    extract_cfg = ExtractConfig(
        batch_size=params.extract_batch_size,
        max_ngram_fallback=params.max_ngram_fallback,
        ticker_noise_filter=params.ticker_noise_filter,
    )
    resources = build_resources(
        tickers_df, secind_df, topics_df, cfg=extract_cfg
    )

    raw = drop_contraction_artifacts(extract_raw_matches(messages, resources))
    if raw.empty:
        return pd.DataFrame()

    groups = build_group_payloads(
        raw, max_candidates_per_msg=params.max_candidates_per_msg
    )
    groups["prompt"] = build_prompts(
        groups["msg"], groups["payload"], params.output_format
    )

    n_cands = groups["payload"].map(len)
    logger.info(
        f"Messages with candidates: {len(groups)} of {len(messages)}, "
        f"candidates per message: median {n_cands.median():.0f}, max {n_cands.max()}"
    )
    return groups


def make_validator(params: NerParams, hf_token: str) -> EntityValidatorLLM:
    return EntityValidatorLLM(
        model_id=params.model_id,
        token=hf_token,
        use_chat_template=params.use_chat_template,
        output_format=params.output_format,
        prefill_answer=params.prefill_answer,
        max_new_tokens=params.max_new_tokens,
        repetition_penalty=params.repetition_penalty,
        batch_size=params.batch_size,
        max_batch_tokens=params.max_batch_tokens,
        max_input_tokens=params.max_input_tokens,
        new_tokens_base=params.new_tokens_base,
        new_tokens_per_candidate=params.new_tokens_per_candidate,
    )


def log_llm_stats(outputs: list) -> None:
    """Quality summary of LLM outputs; anything lost is a WARNING."""
    n = len(outputs)
    reasons = pd.Series([o.get("stop_reason") for o in outputs])
    skipped = sum(o["skipped"] for o in outputs)
    cut = sum(not o["finished"] and not o["skipped"] for o in outputs)
    salvaged = sum(o.get("salvaged", False) for o in outputs)
    lost = sum(
        not o["parse_ok"] and not o.get("salvaged", False) and not o["skipped"]
        for o in outputs
    )
    kept = sum(len(o["entities"]) for o in outputs)
    logger.info(f"LLM outputs: {n}, entities kept: {kept}")
    logger.info(f"Stop reasons: {reasons.value_counts().to_dict()}")
    if skipped:
        logger.warning(
            f"{skipped}/{n} prompts longer than max_input_tokens were skipped"
        )
    if cut:
        logger.warning(
            f"{cut}/{n} answers were cut (token limit or looping), "
            f"{salvaged} recovered from the cut JSON"
        )
    if lost:
        logger.warning(
            f"{lost}/{n} answers could not be parsed, their entities are lost"
        )


def validate_entities(
    groups: pd.DataFrame, params: NerParams, hf_token: str
) -> pd.DataFrame:
    """Run the LLM over all candidate groups and return accepted entities."""
    require_gpu()
    validator = make_validator(params, hf_token)
    outputs = validator.run(
        groups["prompt"].tolist(),
        n_candidates=groups["payload"].map(len).tolist(),
    )
    log_llm_stats(outputs)

    validated = build_validated_df(groups, outputs)
    types = validated["entity_type"].value_counts().to_dict()
    logger.info(f"Accepted entities: {len(validated)}, by type: {types}")
    return validated


def prepare_output(validated: pd.DataFrame) -> pd.DataFrame:
    """Shape accepted entities for the target table."""
    out = validated.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out["msg_date"] = out["ts"].dt.date
    out[["user_id", "chat_id"]] = out[["user_id", "chat_id"]].astype("int64")
    out["msg"] = out["msg"].astype("string").str.slice(0, MSG_MAX_CHARS)

    too_long = (out["entity_id"].str.len() > ENTITY_ID_MAX_CHARS) | (
        out["entity_type"].str.len() > ENTITY_TYPE_MAX_CHARS
    )
    if too_long.any():
        logger.warning(
            f"Dropping {int(too_long.sum())} entities longer than the table columns: "
            f"{out.loc[too_long, 'entity_id'].head(5).tolist()}"
        )
        out = out[~too_long]
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def run(
    settings: Settings,
    params: Optional[NerParams] = None,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
) -> int:
    """Run the whole NER job and return the number of rows loaded.

    dry_run=True does everything except writing to S3/Redshift.
    max_rows limits how many messages go to the LLM (for quick tests).
    """
    params = params or NerParams()
    hf_token = export_hf_token(settings)

    messages = load_messages(settings)
    if messages.empty:
        logger.info("No new messages, nothing to do")
        return 0

    groups = build_candidates(settings, messages, params)
    if groups.empty:
        logger.info("No entity candidates in new messages, nothing to do")
        return 0
    if max_rows:
        groups = groups.head(max_rows)

    validated = validate_entities(groups, params, hf_token)
    if validated.empty:
        logger.info("LLM accepted no entities, nothing to load")
        return 0
    output = prepare_output(validated)

    if dry_run:
        logger.info(f"Dry run: {len(output)} rows prepared, nothing written")
        return len(output)

    return load_df_to_redshift(
        settings,
        output,
        settings.ner_table,
        OUTPUT_COLUMNS,
        S3_FILE_NAME,
    )
