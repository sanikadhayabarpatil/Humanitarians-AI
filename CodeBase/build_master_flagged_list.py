#!/usr/bin/env python3
"""
Stage 1 → Stage 2 Merger (Optimized Version)

Builds a master list of all *flagged* assertions across the pipeline.

A statement is CONSIDERED FLAGGED if:
    - Stage 1 initial verdict is "Refuted" or "Uncertain", OR
    - Stage 2 majority verdict is "Refuted", OR
    - Stage 3 final_decision is "Needs Review"

This makes the flagged list consistent with the optimized pipeline.
"""

import os
import json
import glob
from typing import List, Dict, Any

from logger import logger

OUTPUT_DIR = "output"


# -------------------------------------------------------------
# Helper: Load JSON safely
# -------------------------------------------------------------
def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return default


# -------------------------------------------------------------
# DISCOVER ALL RUN IDS
# -------------------------------------------------------------
def discover_available_runs() -> List[int]:
    """
    Finds all output/fact_check_results_run*.json files
    and extracts the run IDs.
    """
    pattern = os.path.join(OUTPUT_DIR, "fact_check_results_run*.json")
    files = glob.glob(pattern)

    run_ids = []
    for path in files:
        name = os.path.basename(path)
        try:
            run_id = int(name.replace("fact_check_results_run", "").replace(".json", ""))
            run_ids.append(run_id)
        except ValueError:
            continue

    return sorted(set(run_ids))


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def merge_flagged_assertions():
    logger.info("========== BUILDING MASTER FLAGGED ASSERTIONS LIST ==========")

    # Displays all Stage-1 results found
    run_ids = discover_available_runs()
    if not run_ids:
        logger.error("No Stage 1 results found. Run Stage 1 first.")
        return

    logger.info(f"Found Stage-1 runs: {run_ids}")

    master_flagged: Dict[str, Dict[str, Any]] = {}

    # ------------------------------
    # Load Stage-2 aggregated results
    # ------------------------------
    stage2_path = os.path.join(OUTPUT_DIR, "verification_aggregated.json")
    verification = _load_json(stage2_path, [])

    stage2_by_idx = {v.get("index"): v for v in verification}

    # ------------------------------
    # Process all Stage-1 runs
    # ------------------------------
    for run_id in run_ids:
        s1_path = os.path.join(OUTPUT_DIR, f"fact_check_results_run{run_id}.json")
        a1_path = os.path.join(OUTPUT_DIR, f"assertions_run{run_id}.json")

        stage1_results = _load_json(s1_path, [])
        assertions = _load_json(a1_path, [])

        assertion_lookup = {a["index"]: a for a in assertions}

        for row in stage1_results:
            idx = row.get("index")
            if idx is None:
                continue

            initial = row.get("final_verdict", "Uncertain")

            # Stage-2 data (if exists)
            s2 = stage2_by_idx.get(idx, {})
            maj = s2.get("majority_verdict", "Uncertain")
            final_decision = s2.get("final_decision", None)

            # Original assertion text
            original_text = (
                assertion_lookup.get(idx, {}).get("original_statement")
                or row.get("assertion", "")
            ).strip()

            # ------------------------------
            # FLAGGING RULES
            # ------------------------------
            is_flagged = (
                initial in ("Refuted", "Uncertain")
                or maj == "Refuted"
                or final_decision == "Needs Review"
            )

            if not is_flagged:
                continue

            # Deduplicate by text
            key = original_text.lower()
            if not key:
                continue

            master_flagged[key] = {
                "index": idx,
                "original_statement": original_text,
                "initial_verdict": initial,
                "stage2_majority": maj,
                "final_decision": final_decision,
                "reasoning": row.get("reasoning", ""),
                "run_id": run_id,
            }

    # ------------------------------
    # Save outputs
    # ------------------------------
    output_json = os.path.join(OUTPUT_DIR, "master_flagged_assertions.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(list(master_flagged.values()), f, indent=2)

    output_csv = os.path.join(OUTPUT_DIR, "master_flagged_assertions.csv")
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("index,original_statement,initial_verdict,stage2_majority,final_decision\n")
        for item in master_flagged.values():
            stmt = item["original_statement"].replace('"', "'").replace(",", ";")
            f.write(
                f'{item["index"]},"{stmt}",{item["initial_verdict"]},'
                f'{item["stage2_majority"]},{item["final_decision"]}\n'
            )

    logger.info(f"Saved:\n - {output_json}\n - {output_csv}")
    logger.info("========== MASTER FLAGGED LIST COMPLETE ==========")


if __name__ == "__main__":
    merge_flagged_assertions()
