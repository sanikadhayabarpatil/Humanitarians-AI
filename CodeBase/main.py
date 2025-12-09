#!/usr/bin/env python3
"""
Main Orchestrator for the Optimized Scientific Fact-Checking Pipeline

This orchestrates the 3-stage optimized workflow:

STAGE 1  (HEAVY WORK, RUNS ONCE)
    - Read chapter
    - Build local RAG
    - Extract assertions (Gemini)
    - Retrieve evidence (PubMed/Tavily/local RAG)
    - Fact-check ONCE
    OUTPUT:
        output/assertions_run1.json
        output/fact_check_results_run1.json
        output/evidence_store.json

STAGE 2  (LIGHT, 5× VERIFICATION RUNS)
    - Reuse evidence_store.json
    - Re-run reasoning with different random seeds
    - Produce majority verdict, agreement ratio, mean confidence
    OUTPUT:
        output/verification_raw_passes.json
        output/verification_aggregated.json

STAGE 3
    - Merge Stage-1 + Stage-2
    - Produce final_decisions
    - Human summary
    OUTPUT:
        output/final_results.json/.csv
        output/final_summary.txt
"""

import argparse
from pathlib import Path

from logger import logger
from run_stage1 import run_stage1
from run_stage2_verification import run_stage2_verification
from build_master_flagged_list import merge_flagged_assertions
from generate_final_summary import generate_final_summary


# ---------------------------------------------------------------------
# Run the entire optimized pipeline
# ---------------------------------------------------------------------
def run_full_pipeline(chapter_path: str, num_passes: int = 5) -> None:
    chapter_file = Path(chapter_path)

    if not chapter_file.exists():
        logger.error(f"❌ Chapter file not found: {chapter_file}")
        return

    logger.info("========== FULL FACT-CHECKING PIPELINE START ==========")
    logger.info(f"📘 Chapter: {chapter_file}")

    # ---------------------------------------------------------
    # STAGE 1 — heavy processing (run once)
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 1: Running heavy single-pass processing ---")
    run_stage1(str(chapter_file))

    # ---------------------------------------------------------
    # Build flagged assertion list (optional intermediate step)
    # ---------------------------------------------------------
    logger.info("\n--- INTERMEDIATE: Building flagged assertions list ---")
    merge_flagged_assertions()

    # ---------------------------------------------------------
    # STAGE 2 — 5× lightweight reasoning passes
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 2: Multi-Pass Verification (5×) ---")
    run_stage2_verification(num_passes=num_passes)

    # ---------------------------------------------------------
    # STAGE 3 — final merge + summary
    # ---------------------------------------------------------
    logger.info("\n--- STAGE 3: Summary + Final Results ---")
    generate_final_summary()

    logger.info("========== FULL PIPELINE COMPLETE ==========")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full optimized scientific fact-checking pipeline."
    )

    parser.add_argument(
        "--chapter",
        type=str,
        required=True,
        help="Path to the chapter .md/.txt file",
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=5,
        help="Number of Stage-2 verification passes (default = 5)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = _parse_cli_args()
    run_full_pipeline(chapter_path=args.chapter, num_passes=args.passes)
