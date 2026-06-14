"""Deterministic page-citation parsing and validation.

Shared by the API (to filter displayed sources to genuinely cited pages) and by
the answer-level evaluation harness (to flag citations to pages that were never
retrieved). Pure functions, no I/O, fully testable.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# Matches [p. 93], [p.93], [pp. 93, 94], [p 93], [page 93] and lists of numbers.
_CITATION_RE = re.compile(r"\[\s*(?:p{1,2}\.?|pages?)\s*([0-9,\s]+?)\s*\]", re.IGNORECASE)


def extract_citations(text: str) -> List[int]:
    """Return the sorted, unique page numbers cited in ``text``.

    Recognizes [p. N], [pp. N, M], [page N] forms. Returns [] when none.
    """
    if not text:
        return []
    pages = set()
    for body in _CITATION_RE.findall(text):
        for num in re.findall(r"\d+", body):
            pages.add(int(num))
    return sorted(pages)


def validate_citations(
    cited_pages: Sequence[int], retrieved_pages: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Split cited pages into (supported, unsupported).

    A citation is *supported* only if that page was among the pages supplied to
    the generator (``retrieved_pages``). Unsupported citations are a red flag:
    the model cited a page it was never shown.
    """
    available = set(retrieved_pages)
    supported = [p for p in cited_pages if p in available]
    unsupported = [p for p in cited_pages if p not in available]
    return supported, unsupported
