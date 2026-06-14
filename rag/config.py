"""Central configuration for the Laws-of-the-Game RAG retrieval core.

All tunable retrieval parameters live here so the API, the offline evaluation
harness, the notebook and the tests share exactly one source of truth. Values
can be overridden from the environment (handy for Vercel / CI) but every default
is chosen to be safe and deterministic.

This module imports nothing beyond the standard library, so it is cheap to load
at serverless cold start and never drags in FastAPI / httpx.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root = parent of this package directory. Resolved once, absolutely,
# so retrieval works regardless of the current working directory.
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
DEFAULT_CHUNKS_PATH: Path = DATA_DIR / "chunks.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RetrievalConfig:
    """Immutable retrieval configuration.

    Attributes
    ----------
    top_k:
        Number of results returned to the generator / evaluation.
    candidate_pool:
        How many scored candidates to consider before page deduplication.
        Must be >= top_k so deduplication has material to work with.
    bm25_k1, bm25_b:
        Standard BM25 free parameters.
    expand_query:
        Whether colloquial->official query expansion is applied.
    deduplicate_pages:
        When True, keep only the highest-scoring chunk per PDF page so the
        top_k results cover up to top_k distinct pages.
    min_token_len:
        Tokens shorter than this (after lowercasing) are dropped.
    history_turns:
        Max number of recent user turns inspected for conversational rewrite.
    max_query_chars:
        Hard cap on the length of a (possibly rewritten) retrieval query.
    """

    top_k: int = 5
    candidate_pool: int = 30
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    expand_query: bool = True
    deduplicate_pages: bool = True
    min_token_len: int = 2
    history_turns: int = 3
    max_query_chars: int = 400

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be >= top_k")

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Build a config, letting RAG_* environment variables override defaults."""
        base = cls()
        return cls(
            top_k=_env_int("RAG_TOP_K", base.top_k),
            candidate_pool=_env_int("RAG_CANDIDATE_POOL", base.candidate_pool),
            expand_query=_env_bool("RAG_EXPAND_QUERY", base.expand_query),
            deduplicate_pages=_env_bool("RAG_DEDUPLICATE_PAGES", base.deduplicate_pages),
            history_turns=_env_int("RAG_HISTORY_TURNS", base.history_turns),
            max_query_chars=_env_int("RAG_MAX_QUERY_CHARS", base.max_query_chars),
        )


# A frozen default instance most callers can use directly.
DEFAULT_CONFIG = RetrievalConfig()
