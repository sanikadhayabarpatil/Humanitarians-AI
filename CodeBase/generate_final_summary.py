#!/usr/bin/env python3
"""
Stage 3 – Final Summary and Aggregation (Optimized Version)

This consumes:
    • Stage 1 outputs:
          output/assertions_run1.json
          output/fact_check_results_run1.json
    • Stage 2 outputs:
          output/verification_aggregated.json

It merges initial verdicts with multi-pass aggregated reasoning.

Outputs:
    • output/final_results.json
    • output/final_results.csv
    • output/final_summary.txt
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from logger import logger

OUTPUT_DIR = Path("output")


# ------------------------- Helpers -------------------------

def _load_json(path: Path, expected_type):
    if not path.exists():
        logger.error(f"Missing required file: {path}")
        return expected_type()
    try:
        data = json.loads(path.read_text())
        if isinstance(data, expected_type):
            return data
        logger.error(f"{path} did NOT contain a {expected_type.__name__}.")
        return expected_type()
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return expected_type()


def _compute_final_decision(initial_verdict: str,
                            majority_verdict: str,
                            agreement_ratio: float) -> str:
    """
    Rules for final decision:
       If Stage-2 has ≥ 60% majority for Supported     → Supported
       If Stage-2 has ≥ 60% majority for Refuted       → Refuted
       Otherwise                                       → Needs Review
    """
    majority_verdict = majority_verdict or "Uncertain"
    initial_verdict = initial_verdict or "Uncertain"

    strong = 0.6

    if majority_verdict == "Supported" and agreement_ratio >= strong:
        return "Supported"
    if majority_verdict == "Refuted" and agreement_ratio >= strong:
        return "Refuted"

    return "Needs Review"


# ------------------------- Main -------------------------

def generate_final_summary():
    logger.info("========== STAGE 3: Final Summary ==========")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load required inputs
    assertions = _load_json(OUTPUT_DIR / "assertions_run1.json", list)
    stage1 = _load_json(OUTPUT_DIR / "fact_check_results_run1.json", list)
    stage2 = _load_json(OUTPUT_DIR / "verification_aggregated.json", list)

    if not assertions or not stage1 or not stage2:
        logger.error("Missing Stage 1 or Stage 2 data. Cannot run Stage 3.")
        return

    # Build lookup tables
    a_by_idx: Dict[int, Dict] = {a["index"]: a for a in assertions}
    s1_by_idx: Dict[int, Dict] = {r["index"]: r for r in stage1}
    s2_by_idx: Dict[int, Dict] = {r["index"]: r for r in stage2}

    final_records: List[Dict[str, Any]] = []

    all_indices = sorted(a_by_idx.keys())

    for idx in all_indices:
        a = a_by_idx.get(idx, {})
        s1 = s1_by_idx.get(idx, {})
        s2 = s2_by_idx.get(idx, {})

        assertion_text = (
            a.get("original_statement")
            or s1.get("assertion")
            or ""
        )

        initial_verdict = s1.get("final_verdict", "Uncertain")
        initial_conf = float(s1.get("confidence", 0.0))

        majority = s2.get("majority_verdict", "Uncertain")
        agreement = float(s2.get("agreement_ratio", 0.0))
        mean_conf = float(s2.get("mean_confidence", 0.0))

        final_decision = _compute_final_decision(
            initial_verdict,
            majority,
            agreement,
        )

        rec = {
            "index": idx,
            "assertion": assertion_text,
            # Stage 1
            "initial_verdict": initial_verdict,
            "initial_confidence": initial_conf,
            # Stage 2 aggregated
            "majority_verdict": majority,
            "agreement_ratio": agreement,
            "mean_confidence": mean_conf,
            "num_supported": s2.get("num_supported", 0),
            "num_refuted": s2.get("num_refuted", 0),
            "num_uncertain": s2.get("num_uncertain", 0),
            # Final decision
            "final_decision": final_decision,
        }

        final_records.append(rec)

    if not final_records:
        logger.error("No final records created.")
        return

    # ------------------------- Save JSON -------------------------
    json_path = OUTPUT_DIR / "final_results.json"
    json_path.write_text(json.dumps(final_records, indent=2))
    logger.info(f"💾 Saved detailed final results → {json_path}")

    # ------------------------- Save CSV -------------------------
    csv_path = OUTPUT_DIR / "final_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "index,assertion,initial_verdict,initial_confidence,"
            "majority_verdict,agreement_ratio,mean_confidence,"
            "num_supported,num_refuted,num_uncertain,final_decision\n"
        )
        for r in final_records:
            assertion = (r["assertion"] or "").replace('"', "'")
            f.write(
                f'{r["index"]},"{assertion}",'
                f'{r["initial_verdict"]},{r["initial_confidence"]},'
                f'{r["majority_verdict"]},{r["agreement_ratio"]},'
                f'{r["mean_confidence"]},{r["num_supported"]},'
                f'{r["num_refuted"]},{r["num_uncertain"]},'
                f'{r["final_decision"]}\n'
            )
    logger.info(f"💾 Saved CSV → {csv_path}")

    # ------------------------- Human summary -------------------------
    decisions = Counter(r["final_decision"] for r in final_records)
    majority_votes = Counter(r["majority_verdict"] for r in final_records)

    summary_lines = [
        "FINAL FACT-CHECKING SUMMARY",
        "=============================",
        f"Total assertions analyzed: {len(final_records)}",
        "",
        "Final decisions:",
        f"  Supported:    {decisions.get('Supported', 0)}",
        f"  Refuted:      {decisions.get('Refuted', 0)}",
        f"  Needs Review: {decisions.get('Needs Review', 0)}",
        "",
        "Stage-2 Majority Verdicts:",
    ]

    for verdict, count in majority_votes.most_common():
        summary_lines.append(f"  {verdict}: {count}")

    summary_text = "\n".join(summary_lines)

    summary_path = OUTPUT_DIR / "final_summary.txt"
    summary_path.write_text(summary_text)
    logger.info(f"💾 Saved human-readable summary → {summary_path}")

    logger.info("🎉 Stage 3 Complete.")


if __name__ == "__main__":
    generate_final_summary()
