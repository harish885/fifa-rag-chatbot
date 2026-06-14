"""A small, dependency-free BM25 ranking function.

Pure standard library so it builds in well under a cold-start budget for a few
hundred chunks and ships inside a serverless function with zero ML dependencies.
Ranking is deterministic with an explicit tie-break (see ``search``).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Sequence, Tuple


class BM25:
    """Okapi BM25 over a fixed tokenized corpus.

    Parameters
    ----------
    corpus_tokens:
        One token list per document, in corpus order.
    k1, b:
        Standard BM25 free parameters.
    """

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = [len(d) for d in corpus_tokens]
        n = len(corpus_tokens)
        self.n = n
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.tf: List[Counter] = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        # BM25 idf with +1 smoothing keeps idf non-negative for all terms.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document for the given query tokens."""
        if self.avg_len == 0:
            return 0.0
        tf = self.tf[doc_index]
        dl = self.doc_len[doc_index]
        s = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return s

    def search(
        self, query_tokens: Sequence[str], top_k: int = 5
    ) -> List[Tuple[float, int]]:
        """Return up to ``top_k`` ``(score, doc_index)`` pairs, best first.

        Only documents with a strictly positive score are returned. Ties in
        score are broken by **lower document index first** (deterministic and
        independent of dict/iteration order). A non-positive ``top_k`` returns
        an empty list; an empty corpus or empty query also returns ``[]``.
        """
        if top_k <= 0 or self.n == 0 or not query_tokens:
            return []
        scored = [
            (s, i)
            for i in range(self.n)
            if (s := self.score(query_tokens, i)) > 0
        ]
        # Sort: highest score first, then smallest index first.
        scored.sort(key=lambda si: (-si[0], si[1]))
        return scored[:top_k]
