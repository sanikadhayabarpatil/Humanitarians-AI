#!/usr/bin/env python3
"""
Stage 1 – Single-Pass Chapter Fact-Checking (Batched + Optimized)

What this stage does (ONCE per chapter):

1. Read the chapter file.
2. Build a local RAG knowledge base (TF-IDF over chapter chunks).
3. Extract ALL factual assertions from the chapter in a single Gemini call.
4. Retrieve external + local evidence for each assertion.
5. Fact-check all assertions in BATCHES using Gemini (few calls total).
6. Save artifacts for downstream stages:

    output/assertions_run1.json
    output/fact_check_results_run1.json
    output/fact_check_results_run1.csv
    output/evidence_store.json

Stage 2 will reuse assertions_run1.json + evidence_store.json to perform
multi-pass verification without repeating heavy work.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from config import load_config
from logger import logger
from scientific_fact_checker import ScientificFactChecker


def run_stage1(chapter_path: str) -> None:
    """
    Run Stage 1 once for the given chapter:
      - read chapter
      - build RAG knowledge base
      - extract assertions (single heavy pass)
      - batched fact-check of all assertions
      - save results for run_id = 1
    """
    load_dotenv()

    chapter_path_obj = Path(chapter_path)
    if not chapter_path_obj.exists():
        logger.error(f"❌ Chapter file not found: {chapter_path_obj}")
        return

    # Load configuration & initialize checker
    config = load_config()
    checker = ScientificFactChecker(config)

    # Read chapter and build KB once
    chapter_text = checker.read_chapter(str(chapter_path_obj))
    if not chapter_text:
        logger.error("❌ Chapter text is empty. Aborting Stage 1.")
        return

    logger.info("📚 Building RAG knowledge base from chapter text (single pass)...")
    checker.build_knowledge_base(chapter_text)

    # Single run_id for optimized pipeline
    run_id = 1
    pass_id = 1  # first (baseline) reasoning pass

    # Extract assertions once (single heavy Gemini call, batched JSON)
    logger.info("🧩 Extracting assertions (single heavy pass)...")
    assertions = checker.extract_assertions(str(chapter_path_obj), run_id=run_id)
    if not assertions:
        logger.error("❌ No assertions extracted. Nothing to fact-check.")
        return

    logger.info(
        f"✅ Extracted {len(assertions)} assertions. "
        "Starting batched fact-checking for Stage 1..."
    )

    # Batched fact-check of all assertions using the built KB + evidence
    results = checker.fact_check_assertions(
        assertions,
        run_id=run_id,
        pass_id=pass_id,
    )

    if not results:
        logger.error("❌ No fact-checking results produced. Aborting Stage 1.")
        return

    # Persist artifacts for downstream stages
    checker.save_run_results(run_id=run_id, assertions=assertions, results=results)

    logger.info("🎯 Stage 1 complete – single-pass batched fact-checking finished successfully.")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 – Single-pass batched scientific fact-checking for a chapter"
    )
    parser.add_argument(
        "--chapter",
        type=str,
        required=True,
        help="Path to the chapter .md/.txt file to fact-check",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    logger.info("========== STAGE 1: Single-Pass Fact-Checking ==========")
    logger.info(f"Using chapter: {args.chapter}")
    run_stage1(args.chapter)
