"""Sentiment job: classify Ask SA follow-up messages and load to Redshift.

Production version of notebooks/sentiment-analysis.ipynb.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from asksa_follow_up.ask_sa_followup import FollowupLLM, build_followups
from asksa_jobs.common import (
    Settings,
    export_hf_token,
    fetch_df,
    load_df_to_redshift,
    read_query,
)

logger = logging.getLogger(__name__)

QUERY_FILE = "get_chats.sql"
S3_FILE_NAME = "result.csv"
TEXT_MAX_CHARS = 256
PARSE_FAILED = ["no_json", "json_parse_error"]

# Same order as ds.production__ask_sa_sentiment_followup_msg_analysis
OUTPUT_COLUMNS = [
    "user_id",
    "chat_id",
    "followup_number",
    "followup_row_index",
    "followup_time",
    "initial_query",
    "ask_sa_answer",
    "followup_query",
    "label",
    "subtype",
    "confidence",
    "rationale",
    "model_used",
    "raw_model_text",
    "followup_date",
]
TEXT_COLUMNS = [
    "initial_query",
    "ask_sa_answer",
    "followup_query",
    "rationale",
    "raw_model_text",
]


@dataclass(frozen=True)
class SentimentParams:
    """LLM settings for the sentiment job.

    max_input_tokens: raised from 1400. On real data 56% of prompts were
        longer (max ~2400 tokens) and got cut from the right, which removed
        the follow-up message itself.
    max_batch_tokens: rows are length-sorted and batched so that
        rows * longest_prompt <= 5120, i.e. never more than the old worst
        case of 2 rows at full length. Short prompts get up to batch_size
        rows per batch, so GPU memory stays the same and runs are faster.
    """

    model_id: str = "google/gemma-2-9b-it"
    max_input_tokens: int = 2560
    max_new_tokens: int = 96
    batch_size: int = 8
    max_batch_tokens: int = 5120
    enable_json_repair: bool = True


def require_gpu() -> None:
    """Fail fast instead of running a 9B model on CPU for hours."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, refusing to run on CPU")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def load_followups(settings: Settings) -> pd.DataFrame:
    """Fetch new chats and turn them into one row per follow-up message."""
    chats_df = fetch_df(settings, read_query(settings, QUERY_FILE))
    if chats_df.empty:
        return pd.DataFrame()

    followups = build_followups(chats_df)
    logger.info(
        f"Chat rows: {len(chats_df)}, follow-ups extracted: {len(followups)}"
    )
    return followups


def count_truncated(llm: FollowupLLM, followups: pd.DataFrame) -> int:
    """Count prompts longer than llm.max_input_tokens (tokenization only)."""
    lengths = llm.prompt_lengths(followups)
    return sum(length > llm.max_input_tokens for length in lengths)


def score_followups(
    followups: pd.DataFrame, params: SentimentParams, hf_token: str
) -> pd.DataFrame:
    """Classify follow-ups with the LLM and log a quality summary."""
    require_gpu()
    llm = FollowupLLM(
        model_id=params.model_id,
        max_input_tokens=params.max_input_tokens,
        max_new_tokens=params.max_new_tokens,
        batch_size=params.batch_size,
        max_batch_tokens=params.max_batch_tokens,
        token=hf_token,
        enable_json_repair=params.enable_json_repair,
    )

    truncated = count_truncated(llm, followups)
    if truncated:
        logger.warning(
            f"{truncated}/{len(followups)} prompts exceed "
            f"{llm.max_input_tokens} tokens and will be cut from the right "
            "(the follow-up is lost), consider raising max_input_tokens"
        )
    else:
        logger.info(f"All prompts fit into {llm.max_input_tokens} tokens")

    scored = llm.classify_df(followups)

    labels = scored["label"].value_counts(dropna=False).to_dict()
    subtypes = scored["subtype"].value_counts(dropna=False).head(20).to_dict()
    unparsed = int(scored["rationale"].isin(PARSE_FAILED).sum())
    logger.info(f"Label counts: {labels}")
    logger.info(f"Top subtypes: {subtypes}")
    logger.info(f"Rows unparsed after JSON repair: {unparsed}/{len(scored)}")
    return scored


def prepare_output(scored: pd.DataFrame) -> pd.DataFrame:
    """Shape scored rows for the target table."""
    out = scored.copy()
    out["followup_date"] = pd.to_datetime(out["followup_time"]).dt.date
    out[["user_id", "chat_id"]] = out[["user_id", "chat_id"]].astype(int)
    out[TEXT_COLUMNS] = out[TEXT_COLUMNS].apply(
        lambda s: s.astype("string").str.slice(0, TEXT_MAX_CHARS)
    )
    return out[OUTPUT_COLUMNS]


def run(
    settings: Settings,
    params: Optional[SentimentParams] = None,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
) -> int:
    """Run the whole sentiment job and return the number of rows loaded.

    dry_run=True does everything except writing to S3/Redshift.
    max_rows limits how many follow-ups are scored (for quick tests).
    """
    params = params or SentimentParams()
    hf_token = export_hf_token(settings)

    followups = load_followups(settings)
    if followups.empty:
        logger.info("No new follow-ups, nothing to do")
        return 0
    if max_rows:
        followups = followups.head(max_rows)

    output = prepare_output(score_followups(followups, params, hf_token))

    if dry_run:
        logger.info(f"Dry run: {len(output)} rows prepared, nothing written")
        return len(output)

    return load_df_to_redshift(
        settings,
        output,
        settings.sentiment_table,
        OUTPUT_COLUMNS,
        S3_FILE_NAME,
    )
