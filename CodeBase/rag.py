#!/usr/bin/env python3
"""
RAG utilities – SimpleRAGSystem

Optimized for:
- Building a *lightweight* local index over the chapter text.
- Providing short, relevant context chunks for each assertion.
- Avoiding heavy dependencies or complex vector DBs.

Interface used by ScientificFactChecker:
    rag = SimpleRAGSystem()
    rag.build_index_from_text(chapter_text)
    rag.is_index_built() -> bool
    rag.retrieve_chunks(query: str, top_k: int) -> List[dict]

Each returned chunk has the shape:
    {
        "text": str,
        "source_title": str,
        "source_url": Optional[str],
        "metadata": dict
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logger import logger


@dataclass
class TextChunk:
    text: str
    source_title: str = "Chapter"
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SimpleRAGSystem:
    """
    Extremely lightweight RAG index using TF-IDF on chapter chunks.

    Design choices:
    - Single-document index (the current chapter).
    - Fixed-size sliding window chunking.
    - No persistence (rebuilt each Stage 1 run).
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """
        Args:
            chunk_size: approximate number of words per chunk.
            chunk_overlap: number of overlapping words between chunks.
        """
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

        self._chunks: List[TextChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

    # ------------------------------------------------------------------
    # Public API used by ScientificFactChecker
    # ------------------------------------------------------------------
    def build_index_from_text(self, text: str) -> None:
        """
        Chunk the input text and build a TF-IDF index over the chunks.
        """
        text = (text or "").strip()
        if not text:
            logger.error("SimpleRAGSystem.build_index_from_text: empty text.")
            return

        logger.info(
            f"SimpleRAGSystem: chunking chapter with "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

        self._chunks = self._chunk_text(text)
        if not self._chunks:
            logger.error("SimpleRAGSystem: no chunks were created; index not built.")
            return

        docs = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(docs)

        logger.info(
            f"SimpleRAGSystem: built TF-IDF index over {len(self._chunks)} chunks."
        )

    def is_index_built(self) -> bool:
        """
        Returns True if the internal TF-IDF index is ready.
        """
        return self._vectorizer is not None and self._tfidf_matrix is not None

    def retrieve_chunks(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant text chunks for a given query.

        Returns a list of dicts compatible with the ScientificFactChecker's
        expected evidence format.
        """
        if not self.is_index_built():
            logger.warning(
                "SimpleRAGSystem.retrieve_chunks called before index was built."
            )
            return []

        query = (query or "").strip()
        if not query:
            return []

        top_k = max(1, top_k)

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # indices of top_k highest scores
        top_indices = sims.argsort()[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            chunk = self._chunks[int(idx)]
            score = float(sims[int(idx)])

            results.append(
                {
                    "text": chunk.text,
                    "source_title": chunk.source_title,
                    "source_url": chunk.source_url,
                    "score": score,
                    "metadata": chunk.metadata or {},
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> List[TextChunk]:
        """
        Simple word-based sliding window chunking.
        """
        words = text.split()
        n = len(words)
        if n == 0:
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source_title="Chapter",
                        source_url=None,
                        metadata={"start_word": start, "end_word": end},
                    )
                )

            if end == n:
                break
            start += step

        logger.info(f"SimpleRAGSystem: created {len(chunks)} text chunks.")
        return chunks
