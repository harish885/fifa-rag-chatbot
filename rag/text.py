"""Tokenization and stopword handling shared by retrieval and evaluation.

Deliberately dependency-free and deterministic: the same string always yields
the same token list, which keeps BM25 ranking reproducible across runs and
machines.
"""
from __future__ import annotations

import re
from typing import List

# A small, fixed English stopword list. Intentionally hand-curated (not pulled
# from a library) so the behaviour is frozen and auditable for this corpus.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or "
    "s t that the their there these this to was were what when which who will "
    "with within without you your".split()
)

# Tokens are runs of ASCII letters / digits / apostrophes. Apostrophes are kept
# so "player's" tokenizes as a single term rather than splitting awkwardly.
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str, *, min_len: int = 2) -> List[str]:
    """Lowercase, split into word tokens, and drop stopwords / very short tokens.

    Parameters
    ----------
    text:
        Arbitrary input string (may be empty).
    min_len:
        Minimum surviving token length. Tokens shorter than this are removed,
        which discards single letters left over from contractions.
    """
    if not text:
        return []
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in STOPWORDS and len(tok) >= min_len
    ]
