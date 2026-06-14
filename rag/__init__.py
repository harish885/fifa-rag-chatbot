"""Reusable retrieval core for the Laws-of-the-Game RAG chatbot.

This package is the single source of truth for retrieval. The web API
(``api/index.py``), the offline evaluation harness (``eval/``), the notebook and
the tests all import from here, so they exercise identical behaviour. It has no
web-framework dependencies and is cheap to import.
"""
from __future__ import annotations

from .bm25 import BM25
from .citations import extract_citations, validate_citations
from .config import DEFAULT_CONFIG, RetrievalConfig
from .conversation import RetrievalQuery, build_retrieval_query
from .expansion import SYNONYMS, expand_query, matched_phrases
from .models import RetrievalResult
from .retrieval import (
    ChunkValidationError,
    Retriever,
    load_chunks,
    page_ids,
    validate_chunks,
)
from .text import STOPWORDS, tokenize

__all__ = [
    "BM25",
    "RetrievalConfig",
    "DEFAULT_CONFIG",
    "RetrievalQuery",
    "build_retrieval_query",
    "extract_citations",
    "validate_citations",
    "SYNONYMS",
    "expand_query",
    "matched_phrases",
    "RetrievalResult",
    "Retriever",
    "load_chunks",
    "validate_chunks",
    "ChunkValidationError",
    "page_ids",
    "STOPWORDS",
    "tokenize",
]
