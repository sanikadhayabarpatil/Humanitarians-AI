#!/usr/bin/env python3
"""
Centralized configuration loader for the entire fact-checking pipeline.

Why this file exists:
    - Ensures consistent settings across Stage 1, Stage 2, and Stage 3.
    - Prevents each module from trying to read .env separately.
    - Provides SAFE defaults optimized for free-tier Gemini usage.

ALL PARAMETERS HERE ARE OPTIONAL.
If a value exists in the environment (.env), it overrides the defaults.
"""

import os
from dotenv import load_dotenv


# Load environment variables once
load_dotenv()


def load_config() -> dict:
    """
    Returns a dictionary with all relevant configuration values.
    Environment variables override defaults.
    """

    cfg = {}

    # ---------------------------------------------------------
    # GEMINI MODEL + GENERATION SETTINGS
    # ---------------------------------------------------------
    # SAFE FREE-TIER MODELS:
    #   models/gemini-2.5-flash-lite    (fastest, cheapest)
    #   models/gemini-2.5-flash         (better reasoning)
    #
    # We default to flash-lite to avoid quota issues.
    # ---------------------------------------------------------
    cfg["GEMINI_MODEL"] = os.getenv(
        "GEMINI_MODEL",
        "models/gemini-2.5-flash-lite"   # safest free-tier default
    )

    cfg["GEMINI_TEMPERATURE"] = float(os.getenv("GEMINI_TEMPERATURE", 0.2))
    cfg["GEMINI_MAX_TOKENS"] = int(os.getenv("GEMINI_MAX_TOKENS", 4096))

    # ---------------------------------------------------------
    # RAG SETTINGS
    # ---------------------------------------------------------
    cfg["CHUNK_SIZE"] = int(os.getenv("CHUNK_SIZE", 500))
    cfg["CHUNK_OVERLAP"] = int(os.getenv("CHUNK_OVERLAP", 100))

    # ---------------------------------------------------------
    # PUBMED / NCBI SETTINGS
    # ---------------------------------------------------------
    cfg["NCBI_EMAIL"] = os.getenv("PUBMED_EMAIL") or os.getenv("NCBI_EMAIL", "")
    cfg["NCBI_API_KEY"] = os.getenv("NCBI_API_KEY", None)

    # ---------------------------------------------------------
    # TAVILY SEARCH (OPTIONAL)
    # ---------------------------------------------------------
    cfg["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
    cfg["TAVILY_ENDPOINT"] = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")

    # ---------------------------------------------------------
    # SEARCH DOMAINS
    # ---------------------------------------------------------
    cfg["SEARCH_DOMAINS"] = os.getenv(
        "SEARCH_DOMAINS",
        "ncbi.nlm.nih.gov,pubmed.ncbi.nlm.nih.gov"
    ).split(",")

    # ---------------------------------------------------------
    # BATCHING SETTINGS
    # ---------------------------------------------------------
    # (Only used if you later re-enable batching in Stage 1)
    # ---------------------------------------------------------
    cfg["BATCH_SIZE"] = int(os.getenv("BATCH_SIZE", 5))
    cfg["BATCH_DELAY"] = int(os.getenv("BATCH_DELAY", 10))  # secs

    # ---------------------------------------------------------
    # STAGE 2 MULTI-PASS CONFIG
    # ---------------------------------------------------------
    cfg["NUM_VERIFICATION_PASSES"] = int(os.getenv("NUM_VERIFICATION_PASSES", 5))

    # ---------------------------------------------------------
    # SUMMARY SETTINGS
    # ---------------------------------------------------------
    cfg["ENABLE_SUMMARIZATION"] = os.getenv("ENABLE_SUMMARIZATION", "true").lower() == "true"
    cfg["MAX_SUMMARY_CHARS"] = int(os.getenv("MAX_SUMMARY_CHARS", 3000))

    # ---------------------------------------------------------
    # RETURN CONFIG
    # ---------------------------------------------------------
    return cfg
