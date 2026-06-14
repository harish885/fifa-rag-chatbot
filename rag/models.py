"""Structured data types shared across the retrieval core."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk with its provenance and ranking metadata."""

    chunk_id: int
    page: int
    text: str
    score: float
    rank: int  # 1-based position in the returned list

    def to_dict(self) -> dict:
        """JSON-serializable view (rounded score for stable output)."""
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "score": round(self.score, 4),
            "rank": self.rank,
            "text": self.text,
        }

    def preview(self, n: int = 160) -> str:
        """Short single-line snippet for UI source badges."""
        text = " ".join(self.text.split())
        return text if len(text) <= n else text[: n - 1].rstrip() + "…"
