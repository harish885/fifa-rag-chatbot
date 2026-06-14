"""The production retrieval core: load chunks, build BM25, search.

The API server and the offline evaluation harness both construct a ``Retriever``
from this module, guaranteeing they exercise *identical* retrieval behaviour.
Nothing here imports FastAPI / httpx, so it loads cheaply offline and in CI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .bm25 import BM25
from .config import DEFAULT_CHUNKS_PATH, DEFAULT_CONFIG, RetrievalConfig
from .expansion import expand_query, matched_phrases
from .models import RetrievalResult
from .text import tokenize

_REQUIRED_KEYS = {"id", "page", "text"}


class ChunkValidationError(ValueError):
    """Raised when the chunk corpus is missing keys or has invalid values."""


def validate_chunks(chunks: Sequence[dict]) -> None:
    """Validate corpus structure: required keys, positive int pages, non-empty
    text, and globally unique ids. Raises ``ChunkValidationError`` on the first
    structural problem (with the offending index) rather than failing later
    during retrieval.
    """
    if not chunks:
        raise ChunkValidationError("chunk corpus is empty")
    seen_ids = set()
    for i, c in enumerate(chunks):
        missing = _REQUIRED_KEYS - c.keys()
        if missing:
            raise ChunkValidationError(f"chunk {i}: missing keys {sorted(missing)}")
        page = c["page"]
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ChunkValidationError(f"chunk {i}: page must be a positive int, got {page!r}")
        if not isinstance(c["text"], str) or not c["text"].strip():
            raise ChunkValidationError(f"chunk {i}: text must be non-empty")
        cid = c["id"]
        if cid in seen_ids:
            raise ChunkValidationError(f"chunk {i}: duplicate id {cid!r}")
        seen_ids.add(cid)


def load_chunks(path: Optional[Path] = None) -> List[dict]:
    """Read and validate the chunk corpus from JSON."""
    p = Path(path) if path is not None else DEFAULT_CHUNKS_PATH
    if not p.exists():
        raise FileNotFoundError(f"chunk corpus not found: {p}")
    chunks = json.loads(p.read_text(encoding="utf-8"))
    validate_chunks(chunks)
    return chunks


class Retriever:
    """BM25 retriever over the Laws-of-the-Game chunk corpus.

    The index is built once at construction (lightweight for a few hundred
    chunks). ``search`` returns structured, ranked :class:`RetrievalResult`
    objects and applies optional page-level deduplication.
    """

    def __init__(
        self,
        chunks: Optional[Sequence[dict]] = None,
        config: RetrievalConfig = DEFAULT_CONFIG,
        *,
        chunks_path: Optional[Path] = None,
    ) -> None:
        if chunks is None:
            chunks = load_chunks(chunks_path)
        else:
            validate_chunks(chunks)
        self.config = config
        self.chunks: List[dict] = list(chunks)
        self._tokens = [tokenize(c["text"], min_len=config.min_token_len) for c in self.chunks]
        self.index = BM25(self._tokens, k1=config.bm25_k1, b=config.bm25_b)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        expand: Optional[bool] = None,
        deduplicate_pages: Optional[bool] = None,
    ) -> List[RetrievalResult]:
        """Retrieve ranked results for ``query``.

        Per-call overrides fall back to the instance config. When page
        deduplication is on, a deeper candidate pool is scored first and only
        the highest-scoring chunk per page is kept, yielding up to ``top_k``
        distinct pages while preserving score order.
        """
        cfg = self.config
        k = cfg.top_k if top_k is None else top_k
        do_expand = cfg.expand_query if expand is None else expand
        do_dedup = cfg.deduplicate_pages if deduplicate_pages is None else deduplicate_pages

        effective_query = expand_query(query) if do_expand else query.strip().lower()
        q_tokens = tokenize(effective_query, min_len=cfg.min_token_len)
        if not q_tokens:
            return []

        pool = max(cfg.candidate_pool, k) if do_dedup else k
        scored = self.index.search(q_tokens, top_k=pool)

        if do_dedup:
            chosen: List[tuple] = []
            seen_pages = set()
            for score, idx in scored:  # already best-first
                page = self.chunks[idx]["page"]
                if page in seen_pages:
                    continue
                seen_pages.add(page)
                chosen.append((score, idx))
                if len(chosen) >= k:
                    break
            scored = chosen
        else:
            scored = scored[:k]

        return [
            RetrievalResult(
                chunk_id=self.chunks[idx]["id"],
                page=self.chunks[idx]["page"],
                text=self.chunks[idx]["text"],
                score=score,
                rank=rank,
            )
            for rank, (score, idx) in enumerate(scored, start=1)
        ]

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self, query: str, results: Sequence[RetrievalResult]) -> dict:
        """Lightweight retrieval signals used for confidence / refusal logic.

        Returns top score, score gap to the runner-up, number of results, the
        count of unique query terms that matched the corpus vocabulary, and the
        synonym phrases that fired. No hidden state; safe to log.
        """
        q_tokens = tokenize(expand_query(query), min_len=self.config.min_token_len)
        matched_terms = {t for t in q_tokens if t in self.index.idf}
        top = results[0].score if results else 0.0
        second = results[1].score if len(results) > 1 else 0.0
        return {
            "num_results": len(results),
            "top_score": round(top, 4),
            "score_gap": round(top - second, 4),
            "unique_matched_terms": len(matched_terms),
            "expansion_phrases": matched_phrases(query),
        }


def page_ids(results: Iterable[RetrievalResult]) -> List[int]:
    """Convenience: ordered list of pages from a result set."""
    return [r.page for r in results]
