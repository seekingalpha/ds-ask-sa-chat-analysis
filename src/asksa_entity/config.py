from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractConfig:
    # matching
    max_ngram_fallback: int = 6
    context_window: int = 60
    batch_size: int = 5000

    # topics fuzzy
    topic_fuzzy_min_words: int = 4
    topic_rare_df_threshold: int = 3
    topic_fuzzy_min_overlap: int = 2
    topic_fuzzy_score_threshold: float = 0.75
    token_min_len_for_index: int = 3

    # drop obvious ticker false positives before the LLM (see ticker_noise.py)
    ticker_noise_filter: bool = True

    # LLM
    llm_batch_size: int = 1
    llm_max_new_tokens: int = 30
    llm_repetition_penalty: float = 1.05
