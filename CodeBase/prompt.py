#!/usr/bin/env python3
"""
Prompt builders for the Scientific Fact Checking pipeline.

These functions are used by ScientificFactChecker:

    - build_assertion_extraction_prompt(chapter_name, content)
    - build_fact_checking_prompt(assertion, evidence_docs)

The prompts are designed so that Gemini can respond with STRICT JSON that
`scientific_fact_checker.py` can parse directly.
"""

from typing import List, Dict, Any


# ---------------------------------------------------------------------
# Assertion Extraction Prompt
# ---------------------------------------------------------------------
def build_assertion_extraction_prompt(chapter_name: str, content: str) -> str:
    """
    Build a prompt asking Gemini to extract factual scientific assertions
    from the chapter text.

    Expected STRICT JSON output (NO prose, NO markdown):

    [
      {
        "index": 0,
        "original_statement": "string",
        "section": "optional string or null",
        "sentence_number": 12
      },
      ...
    ]

    Notes:
    - `index` should be 0-based and contiguous.
    - `sentence_number` can approximate the sentence position if exact
      detection is hard.
    """
    chapter_name = chapter_name or "Chapter"
    content = content or ""

    prompt = f"""
You are a meticulous scientific assistant.

Your task is to read the following chapter and extract concise factual
assertions that can be objectively checked against the biomedical and
scientific literature.

The chapter is called: "{chapter_name}"

---------------- CHAPTER START ----------------
{content}
----------------- CHAPTER END -----------------

INSTRUCTIONS:

1. Identify individual factual statements that are:
   - Specific enough to be verified or refuted.
   - Primarily about scientific, biomedical, or technical claims.

2. For each assertion you output:
   - Make it self-contained: someone reading only that sentence should
     understand what is being claimed.
   - Avoid vague or purely qualitative claims (e.g., "this is very important").
   - Do NOT include your own commentary or explanation.

3. Output format (VERY IMPORTANT):
   - You MUST output a **strict JSON array** ONLY.
   - Do NOT include any prose, comments, or additional keys.
   - Each element in the array MUST have the following keys:

        "index": integer, 0-based and contiguous.
        "original_statement": string, the concise factual assertion.
        "section": string or null, a coarse section label if you can infer it
                   (e.g., "Introduction", "Methods", "Results", "Discussion").
        "sentence_number": integer, an approximate sentence index (0-based or 1-based,
                           but use the same convention consistently across all items).

4. Examples of desired JSON structure (syntactic shape only):

   [
     {{
       "index": 0,
       "original_statement": "Aspirin reduces the risk of secondary heart attacks when taken daily at low doses.",
       "section": "Introduction",
       "sentence_number": 5
     }},
     {{
       "index": 1,
       "original_statement": "Type 2 diabetes is characterized by insulin resistance in peripheral tissues.",
       "section": "Background",
       "sentence_number": 12
     }}
   ]

5. DO NOT:
   - Do NOT wrap the JSON in markdown backticks.
   - Do NOT add any extra keys.
   - Do NOT include explanations, reasoning, or commentary.

Now carefully read the chapter and produce the JSON array of assertions.
"""
    return prompt.strip()


# ---------------------------------------------------------------------
# Fact Checking Prompt
# ---------------------------------------------------------------------
def build_fact_checking_prompt(
    assertion: str,
    evidence_docs: List[Dict[str, Any]],
) -> str:
    """
    Build a prompt asking Gemini to fact-check a single assertion
    using the provided evidence documents.

    Expected STRICT JSON output (NO prose, NO markdown):

    {
      "final_verdict": "Supported" | "Refuted" | "Uncertain",
      "reasoning": "string explanation",
      "confidence": 0.0-1.0
    }

    Where:
      - "Supported": evidence strongly backs the assertion.
      - "Refuted": evidence contradicts the assertion or shows it is false.
      - "Uncertain": evidence is mixed, weak, or insufficient.
    """
    assertion = assertion or ""
    evidence_docs = evidence_docs or []

    # Build a readable evidence section
    evidence_str_parts = []
    for i, doc in enumerate(evidence_docs, start=1):
        title = doc.get("title") or "Untitled"
        url = doc.get("url") or ""
        snippet = doc.get("snippet") or ""
        source = doc.get("source") or "unknown"

        evidence_str_parts.append(
            f"DOCUMENT {i}:\n"
            f"  Source: {source}\n"
            f"  Title: {title}\n"
            f"  URL: {url}\n"
            f"  Snippet: {snippet}\n"
        )

    evidence_block = "\n".join(evidence_str_parts)

    prompt = f"""
You are a rigorous scientific fact-checking system.

Your job is to evaluate the truth of a single scientific or biomedical
assertion using the provided evidence documents.

---------------- ASSERTION TO CHECK ----------------
{assertion}
----------------- EVIDENCE DOCUMENTS ----------------
{evidence_block}
----------------- END OF EVIDENCE -------------------

INSTRUCTIONS:

1. Carefully read the assertion and all evidence documents.
2. Determine whether the assertion is:

   - "Supported": Evidence consistently and directly supports the assertion.
   - "Refuted": Evidence contradicts the assertion or shows it is false.
   - "Uncertain": Evidence is mixed, weak, indirect, or insufficient to make
                  a confident judgment either way.

3. Provide:
   - A brief but precise textual reasoning that references key points from
     the evidence (e.g., which document(s) support or contradict the claim).
   - A confidence score between 0.0 and 1.0 indicating how strongly the
     evidence justifies your verdict.

4. Output format (VERY IMPORTANT):
   - You MUST output a **single strict JSON object** ONLY.
   - Do NOT include any prose, comments, or markdown.
   - The JSON MUST have exactly these keys:

       "final_verdict": "Supported" | "Refuted" | "Uncertain",
       "reasoning": string,
       "confidence": number between 0.0 and 1.0

5. Examples of desired JSON structure (syntactic shape only):

   {{
     "final_verdict": "Supported",
     "reasoning": "Multiple randomized controlled trials in the provided evidence show a statistically significant reduction in risk, directly matching the population and intervention described in the assertion.",
     "confidence": 0.87
   }}

6. DO NOT:
   - Do NOT wrap the JSON in markdown backticks.
   - Do NOT add extra keys.
   - Do NOT include explanations outside the JSON object.

Now, using ONLY the given evidence, produce the JSON verdict.
"""
    return prompt.strip()
