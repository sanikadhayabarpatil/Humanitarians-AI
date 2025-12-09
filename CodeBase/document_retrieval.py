import os
from typing import List, Dict, Any, Optional

import requests
from Bio import Entrez

from logger import logger


class DocumentRetriever:
    """
    Lightweight biomedical / scientific document retriever.

    Design goals for optimized pipeline:
    - Use *few but strong* sources (PubMed + Tavily) rather than many.
    - Return a small, high-quality set of evidence docs per assertion.
    - Keep interface simple: `search_most_relevant(query, max_results)`.

    Returned document schema (per item):
        {
            "title": str,
            "url": str or None,
            "snippet": str,
            "source": str,           # e.g., "pubmed", "tavily"
            "metadata": dict,        # optional extra fields (pmid, doi, etc.)
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.email: str = config.get("NCBI_EMAIL") or os.getenv("NCBI_EMAIL", "")
        self.ncbi_api_key: Optional[str] = (
            config.get("NCBI_API_KEY") or os.getenv("NCBI_API_KEY")
        )
        self.tavily_api_key: Optional[str] = (
            config.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
        )

        if not self.email:
            logger.warning(
                "NCBI_EMAIL not set – PubMed requests may be rate-limited. "
                "Set NCBI_EMAIL in your environment or config."
            )

        Entrez.email = self.email or "anonymous@example.com"
        if self.ncbi_api_key:
            Entrez.api_key = self.ncbi_api_key

        self.tavily_endpoint: str = (
            config.get("TAVILY_ENDPOINT")
            or os.getenv("TAVILY_ENDPOINT")
            or "https://api.tavily.com/search"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search_most_relevant(
        self, query: str, max_results: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Main entry point used by the fact-checker.

        Strategy:
        1. Query PubMed for biomedical / scientific literature.
        2. If still under max_results, query Tavily for high-quality web docs.
        3. Deduplicate by URL/title and truncate to `max_results`.
        """
        query = (query or "").strip()
        if not query:
            return []

        all_docs: List[Dict[str, Any]] = []

        # 1) PubMed
        try:
            pubmed_docs = self._search_pubmed(query, max_results=max_results)
            all_docs.extend(pubmed_docs)
        except Exception as e:
            logger.error(f"PubMed retrieval failed: {e}")

        # 2) Tavily (only if we still need more and API key exists)
        if len(all_docs) < max_results and self.tavily_api_key:
            try:
                remaining = max_results - len(all_docs)
                tavily_docs = self._search_tavily(query, max_results=remaining)
                all_docs.extend(tavily_docs)
            except Exception as e:
                logger.error(f"Tavily retrieval failed: {e}")

        # Deduplicate by URL + title
        unique: Dict[str, Dict[str, Any]] = {}
        for doc in all_docs:
            key = f"{(doc.get('url') or '').strip().lower()}|{(doc.get('title') or '').strip().lower()}"
            if not key.strip("|"):
                # Both url and title empty – skip
                continue
            if key not in unique:
                unique[key] = doc

        docs = list(unique.values())
        if len(docs) > max_results:
            docs = docs[:max_results]

        logger.info(
            f"Retrieved {len(docs)} total evidence docs "
            f"(requested max {max_results}) for query: {query[:80]}..."
        )
        return docs

    # ------------------------------------------------------------------
    # PubMed
    # ------------------------------------------------------------------
    def _search_pubmed(
        self, query: str, max_results: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Simple PubMed search + fetch metadata + abstract text.

        We keep this intentionally conservative to avoid NCBI overload
        and rely on their recommended email+API key usage.
        """
        logger.info(
            f"📚 PubMed search for '{query[:80]}...' (max_results={max_results})"
        )

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            logger.error(f"PubMed esearch failed: {e}")
            return []

        id_list = record.get("IdList", [])
        if not id_list:
            logger.info("PubMed returned no results.")
            return []

        try:
            fetch_handle = Entrez.efetch(
                db="pubmed", id=",".join(id_list), rettype="medline", retmode="text"
            )
            medline_text = fetch_handle.read()
            fetch_handle.close()
        except Exception as e:
            logger.error(f"PubMed efetch failed: {e}")
            return []

        # Very lightweight parsing: split on blank lines, extract TI/AB/PMID
        docs: List[Dict[str, Any]] = []
        for entry in medline_text.strip().split("\n\n"):
            lines = entry.splitlines()
            pmid = None
            title_parts: List[str] = []
            abstract_parts: List[str] = []

            for line in lines:
                if line.startswith("PMID- "):
                    pmid = line[len("PMID- ") :].strip()
                elif line.startswith("TI  - "):
                    title_parts.append(line[len("TI  - ") :].strip())
                elif line.startswith("AB  - "):
                    abstract_parts.append(line[len("AB  - ") :].strip())
                elif line.startswith("      "):
                    # continuation line
                    if title_parts and not abstract_parts:
                        title_parts.append(line.strip())
                    elif abstract_parts:
                        abstract_parts.append(line.strip())

            title = " ".join(title_parts).strip()
            abstract = " ".join(abstract_parts).strip()

            if not title and not abstract:
                continue

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            snippet = abstract or title

            docs.append(
                {
                    "title": title or "PubMed article",
                    "url": url,
                    "snippet": snippet,
                    "source": "pubmed",
                    "metadata": {
                        "pmid": pmid,
                    },
                }
            )

        logger.info(f"PubMed returned {len(docs)} candidate docs.")
        return docs

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------
    def _search_tavily(
        self, query: str, max_results: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Query Tavily for high-quality web evidence.

        This is used as a *secondary* source after PubMed.
        """
        if not self.tavily_api_key:
            return []

        logger.info(
            f"🌐 Tavily search for '{query[:80]}...' (max_results={max_results})"
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        }

        try:
            resp = requests.post(
                self.tavily_endpoint, json=payload, headers=headers, timeout=20
            )
        except Exception as e:
            logger.error(f"Tavily request failed: {e}")
            return []

        if resp.status_code != 200:
            logger.error(f"Tavily returned HTTP {resp.status_code}: {resp.text[:200]}")
            return []

        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to parse Tavily JSON: {e}")
            return []

        items = data.get("results") or data.get("data") or []
        docs: List[Dict[str, Any]] = []

        for item in items:
            title = item.get("title") or "Web result"
            url = item.get("url")
            snippet = (
                item.get("content")
                or item.get("snippet")
                or item.get("description")
                or ""
            )
            source = item.get("source") or "tavily"

            docs.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": source,
                    "metadata": {},
                }
            )

        logger.info(f"Tavily returned {len(docs)} candidate docs.")
        return docs
