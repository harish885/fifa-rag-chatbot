from rag.bm25 import BM25

CORPUS = [
    ["offside", "position", "opponents", "half"],
    ["handball", "offence", "hand", "arm"],
    ["offside", "offence", "free", "kick"],
]


def test_deterministic_ranking():
    idx = BM25(CORPUS)
    r1 = idx.search(["offside"], top_k=3)
    r2 = idx.search(["offside"], top_k=3)
    assert r1 == r2


def test_no_hit_query_returns_empty():
    idx = BM25(CORPUS)
    assert idx.search(["penalty"], top_k=5) == []


def test_empty_query_returns_empty():
    idx = BM25(CORPUS)
    assert idx.search([], top_k=5) == []


def test_empty_corpus_returns_empty():
    idx = BM25([])
    assert idx.search(["offside"], top_k=5) == []


def test_top_k_bounds():
    idx = BM25(CORPUS)
    assert len(idx.search(["offside"], top_k=1)) == 1
    assert idx.search(["offside"], top_k=0) == []


def test_tie_break_prefers_lower_index():
    # Two identical docs tie on score; the lower index must come first.
    idx = BM25([["foul"], ["foul"]])
    res = idx.search(["foul"], top_k=2)
    assert [i for _, i in res] == [0, 1]


def test_only_positive_scores_returned():
    idx = BM25(CORPUS)
    res = idx.search(["offside", "penalty"], top_k=5)
    assert all(s > 0 for s, _ in res)
