from __future__ import annotations

import pandas as pd

# Topics list taken from your notebook (moved into module).
TOPICS = [
    "2026-outlook",
    "ai-tech-stocks",
    "biotech-stocks",
    "dividends",
    "dividends::dividend-ideas",
    "dividends::dividend-quick-picks",
    "dividends::dividend-strategy",
    "dividends::reits",
    "editors-picks",
    "electric-vehicles",
    "etfs-and-funds",
    "etfs-and-funds::closed-end-funds",
    "etfs-and-funds::etf-analysis",
    "etfs-and-funds::mutual-funds",
    "growth-stocks",
    "interest-rates-stocks",
    "investing-strategy",
    "investing-strategy::bonds",
    "investing-strategy::financial-advisors",
    "investing-strategy::fixed-income",
    "investing-strategy::portfolio-strategy",
    "investing-strategy::retirement",
    "magnificent-7-stocks",
    "market-outlook",
    "market-outlook::commodities",
    "market-outlook::cryptocurrency",
    "market-outlook::forex",
    "market-outlook::gold-and-precious-metals",
    "market-outlook::us-economy",
    "micro-cap-stocks",
    "sectors::communication-services",
    "sectors::consumer-staples",
    "sectors::energy",
    "sectors::real-estate",
    "small-cap-stocks",
    "stock-ideas",
    "stock-ideas::basic-materials",
    "stock-ideas::consumer-goods",
    "stock-ideas::financial",
    "stock-ideas::healthcare",
    "stock-ideas::industrial-goods",
    "stock-ideas::ipos",
    "stock-ideas::long-ideas",
    "stock-ideas::quick-picks",
    "stock-ideas::technology",
    "stock-ideas::utilities",
    "tariffs",
    "undercovered-stocks",
    "us-economy",
    "value-stocks",
    "weight-loss-stocks",
]


def get_topics_df(topics: list[str] | None = None) -> pd.DataFrame:
    """
    Return DataFrame with schema: topic_name
    """
    if topics is None:
        topics = TOPICS
    return pd.DataFrame({"topic_name": list(topics)})
