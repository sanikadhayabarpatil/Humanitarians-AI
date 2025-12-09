#!/usr/bin/env python3
"""
Stage 2 — Multi-Pass Verification (BATCH MODE, FIXED FOR NEW GEMINI SDK)

Changes:
- ONLY 1 Gemini call per verification pass (free-tier safe)
- NO `seed` inside generation_config (SDK removed that field)
- Instead, we add randomness using temperature only
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

from logger import logger
from scientific_fact_checker import ScientificFactChecker

OUTPUT_DIR = Path("output")
ASSERTIONS_PATH = OUTPUT_DIR / "assertions_run1.json"
EVIDENCE_PATH = OUTPUT_DIR / "evidence_store.json"

RAW_OUTPUT = OUTPUT_DIR / "verification_raw_passes.json"
AGG_OUTPUT = OUTPUT_DIR / "verification_aggregated.json"


# -------------------------------------------------------------------------
# Load helper
# -------------------------------------------------------------------------
def _load_json(path: Path, default):
    if not path.exists():
        logger.error(f"Missing file: {path}")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
        return default


# -------------------------------------------------------------------------
# Build batch prompt
# -------------------------------------------------------------------------
def _build_batch_prompt(assertions, evidence_store):

    items = []
    for a in assertions:
        idx = a["index"]
        stmt = a["original_statement"]
        ev = evidence_store.get(str(idx), {}).get("evidence_docs", [])

        items.append({
            "index": idx,
            "assertion": stmt,
            "evidence": ev
        })

    prompt = f"""
You are a scientific fact-checking system.

You will receive an array of objects. Each contains:
- "index"
- "assertion"
- "evidence": list of evidence documents

For EACH item, output an object with EXACT keys:
- "index"
- "final_verdict": Supported | Refuted | Uncertain
- "reasoning"
- "confidence": number 0.0–1.0

Output MUST be a strict JSON array.

INPUT ARRAY:
{json.dumps(items, indent=2)}
"""
    return prompt.strip()


# -------------------------------------------------------------------------
# Stage 2 — Batched verification
# -------------------------------------------------------------------------
def run_stage2_verification(num_passes: int = 5):
    logger.info("========== STAGE 2: MULTI-PASS VERIFICATION (BATCH MODE) ==========")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    assertions = _load_json(ASSERTIONS_PATH, [])
    evidence_store = _load_json(EVIDENCE_PATH, {})

    if not assertions:
        logger.error("❌ No assertions found. Run Stage 1 first.")
        return
    if not evidence_store:
        logger.error("❌ No evidence_store.json found. Run Stage 1 first.")
        return

    checker = ScientificFactChecker(config={})
    logger.info(f"🔁 Running {num_passes} batched passes...")

    all_pass_results: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # EXECUTE 5 PASSES — ONE GEMINI CALL PER PASS
    # ---------------------------------------------------------
    for p in range(1, num_passes + 1):

        logger.info(f"\n🔎 Verification Pass {p}/{num_passes}")

        prompt = _build_batch_prompt(assertions, evidence_store)

        # This is the ONLY Gemini call per pass
        response = checker.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.55,  # drives randomness in new SDK
                "response_mime_type": "application/json",
            }
        )

        try:
            results = json.loads(response.text)
        except Exception as e:
            logger.error(f"❌ Pass {p}: Failed to parse JSON: {e}")
            continue

        for r in results:
            r["pass_id"] = p
            all_pass_results.append(r)

    # ---------------------------------------------------------
    # SAVE RAW RESULTS
    # ---------------------------------------------------------
    RAW_OUTPUT.write_text(json.dumps(all_pass_results, indent=2))
    logger.info(f"💾 Saved raw verification results → {RAW_OUTPUT}")

    # ---------------------------------------------------------
    # AGGREGATE RESULTS
    # ---------------------------------------------------------
    logger.info("📊 Aggregating verification results...")

    grouped: Dict[int, List[Dict]] = {}
    for r in all_pass_results:
        grouped.setdefault(r["index"], []).append(r)

    aggregated = []

    for idx, results in grouped.items():
        verdicts = [r["final_verdict"] for r in results]
        confs = [r["confidence"] for r in results]

        # majority vote
        try:
            from statistics import mode
            majority = mode(verdicts)
        except Exception:
            majority = "Uncertain"

        agreement_ratio = verdicts.count(majority) / len(verdicts)

        aggregated.append({
            "index": idx,
            "majority_verdict": majority,
            "agreement_ratio": round(agreement_ratio, 4),
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "num_supported": verdicts.count("Supported"),
            "num_refuted": verdicts.count("Refuted"),
            "num_uncertain": verdicts.count("Uncertain"),
        })

    AGG_OUTPUT.write_text(json.dumps(aggregated, indent=2))
    logger.info(f"💾 Saved aggregated verification results → {AGG_OUTPUT}")

    logger.info("========== STAGE 2 COMPLETE ==========")


# -------------------------------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    run_stage2_verification(args.passes)
