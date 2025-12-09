#!/usr/bin/env python3
"""
Scientific Fact Checker – Google AI Studio Compatible Version
Supports:
- Free tier Gemini models (recommended: gemini-flash-lite-latest)
- Stage 1 heavy batch fact checking
- Stage 2 lightweight repeated reasoning with fact_check_single_pass()
"""

import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from logger import logger
from rag import SimpleRAGSystem
from document_retrieval import DocumentRetriever
from prompt import (
    build_assertion_extraction_prompt,
    build_fact_checking_prompt,
)

# Google AI Studio SDK
import google.generativeai as genai

OUTPUT_DIR = Path("output")
EVIDENCE_STORE_PATH = OUTPUT_DIR / "evidence_store.json"


class ScientificFactChecker:
    def __init__(self, config: Dict[str, Any]):
        load_dotenv()

        self.config = config or {}

        # FREE TIER safe model
        self.model_name = os.getenv("GEMINI_MODEL", "models/gemini-flash-lite-latest")

        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not api_key:
            raise RuntimeError("❌ Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env")

        genai.configure(api_key=api_key)

        # Gemini model instance
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Using Gemini model: {self.model_name}")

        # Temperature for Stage 1
        self.temperature = float(config.get("GEMINI_TEMPERATURE", 0.2))

        # Evidence limits
        self.max_evidence_docs = int(config.get("MAX_EVIDENCE_DOCS", 8))

        # Retrieval systems
        self.retriever = DocumentRetriever(config)
        self.rag = SimpleRAGSystem()

        self.chapter_text = None
        self.evidence_store = {}

        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # ----------------------------------------------------------------------
    # STAGE 1 — Chapter reading + RAG
    # ----------------------------------------------------------------------
    def read_chapter(self, chapter_path: str) -> str:
        path = Path(chapter_path)
        text = path.read_text(encoding="utf-8")
        self.chapter_text = text
        logger.info(f"Loaded chapter ({len(text.split())} words).")
        return text

    def build_knowledge_base(self, text: Optional[str] = None) -> None:
        self.rag.build_index_from_text(text or self.chapter_text)
        logger.info("RAG knowledge base built.")

    # ----------------------------------------------------------------------
    # STAGE 1 — Assertion Extraction
    # ----------------------------------------------------------------------
    def extract_assertions(self, chapter_path: str, run_id: int = 1):
        chapter_file = Path(chapter_path)
        text = chapter_file.read_text(encoding="utf-8")

        prompt = build_assertion_extraction_prompt(
            chapter_name=chapter_file.name,
            content=text,
        )

        logger.info(f"[Run {run_id}] Extracting assertions...")

        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 8192,
                    },
                )

                data = json.loads(response.text)

                if not isinstance(data, list):
                    raise ValueError("Expected JSON list of assertions")

                assertions = []
                for i, item in enumerate(data):
                    stmt = item.get("original_statement") or item.get("statement")
                    if stmt:
                        assertions.append({
                            "index": i,
                            "original_statement": stmt.strip(),
                            "section": item.get("section"),
                            "sentence_number": item.get("sentence_number", i),
                        })

                logger.info(f"Extracted {len(assertions)} assertions.")
                return assertions

            except Exception as e:
                logger.error(f"Attempt {attempt+1}/3 failed: {e}")

        logger.error("❌ Failed to extract assertions.")
        return []

    # ----------------------------------------------------------------------
    # STAGE 1 — Evidence retrieval
    # ----------------------------------------------------------------------
    def _retrieve_evidence_for_assertion(self, assertion: str):
        external = self.retriever.search_most_relevant(
            assertion, max_results=self.max_evidence_docs
        )

        local = []
        if self.rag.is_index_built():
            chunks = self.rag.retrieve_chunks(assertion, top_k=3)
            for ch in chunks:
                local.append({
                    "title": "Chapter Context",
                    "url": None,
                    "snippet": ch["text"],
                    "source": "local",
                })

        docs = external + local
        return docs[: self.max_evidence_docs]

    # ----------------------------------------------------------------------
    # UNIVERSAL FACT CHECKER — USED BY BOTH STAGE 1 AND STAGE 2
    # ----------------------------------------------------------------------
    def fact_check_single_pass(
        self,
        assertion: str,
        evidence_docs: List[Dict[str, Any]],
        run_id: int,
        pass_id: int,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        This is the core lightweight reasoning function.
        Stage 1 calls it ONCE.
        Stage 2 calls it N times with different seeds.
        """
        if seed is not None:
            random.seed(seed)

        prompt = build_fact_checking_prompt(assertion, evidence_docs)

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 4096,
            },
        )

        try:
            data = json.loads(response.text)
        except Exception:
            return {
                "assertion": assertion,
                "final_verdict": "Uncertain",
                "reasoning": response.text,
                "confidence": 0.0,
                "run_id": run_id,
                "pass_id": pass_id,
            }

        return {
            "assertion": assertion,
            "final_verdict": data.get("final_verdict", "Uncertain"),
            "reasoning": data.get("reasoning", ""),
            "confidence": float(data.get("confidence", 0.0)),
            "run_id": run_id,
            "pass_id": pass_id,
        }

    # ----------------------------------------------------------------------
    # STAGE 1 — heavy pass over ALL assertions
    # ----------------------------------------------------------------------
    def fact_check_assertions(self, assertions, run_id=1):
        results = []
        self.evidence_store = {}

        for item in assertions:
            idx = item["index"]
            stmt = item["original_statement"]

            evidence = self._retrieve_evidence_for_assertion(stmt)
            self.evidence_store[str(idx)] = {
                "assertion": stmt,
                "evidence_docs": evidence,
            }

            result = self.fact_check_single_pass(
                stmt, evidence, run_id=run_id, pass_id=1
            )
            result["index"] = idx
            results.append(result)

        self._save_evidence_store()
        return results

    def _save_evidence_store(self):
        with open(EVIDENCE_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.evidence_store, f, indent=2)
        logger.info(f"Saved evidence store → {EVIDENCE_STORE_PATH}")

    @staticmethod
    def load_evidence_store():
        if not EVIDENCE_STORE_PATH.exists():
            logger.error("Evidence store missing. Run Stage 1 first.")
            return {}
        return json.loads(EVIDENCE_STORE_PATH.read_text())

    def save_run_results(self, run_id, assertions, results):
        OUTPUT_DIR.mkdir(exist_ok=True)

        with open(OUTPUT_DIR / f"assertions_run{run_id}.json", "w") as f:
            json.dump(assertions, f, indent=2)

        with open(OUTPUT_DIR / f"fact_check_results_run{run_id}.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Stage 1 outputs saved.")
