from dataclasses import replace

import pytest

from rag import DEFAULT_CONFIG, ChunkValidationError, Retriever, validate_chunks

CHUNKS = [
    {"id": 0, "page": 10, "text": "Offside position is defined in Law 11."},
    {"id": 1, "page": 10, "text": "An offside offence is penalised with an indirect free kick."},
    {"id": 2, "page": 11, "text": "Offside does not apply from a goal kick."},
    {"id": 3, "page": 12, "text": "Handball offence and the arm position."},
]


def test_result_structure():
    r = Retriever(CHUNKS, replace(DEFAULT_CONFIG, candidate_pool=10))
    res = r.search("offside", top_k=2)
    assert res and res[0].rank == 1
    first = res[0]
    assert {first.chunk_id, first.page} and isinstance(first.score, float)
    assert first.text


def test_page_deduplication_unique_pages():
    cfg = replace(DEFAULT_CONFIG, candidate_pool=10, deduplicate_pages=True)
    res = Retriever(CHUNKS, cfg).search("offside", top_k=3)
    pages = [x.page for x in res]
    assert len(pages) == len(set(pages))  # no duplicate pages
    assert 10 in pages and 11 in pages


def test_no_dedup_allows_same_page_twice():
    cfg = replace(DEFAULT_CONFIG, candidate_pool=10, deduplicate_pages=False)
    res = Retriever(CHUNKS, cfg).search("offside", top_k=3)
    assert [x.page for x in res].count(10) == 2


def test_score_order_preserved_after_dedup():
    cfg = replace(DEFAULT_CONFIG, candidate_pool=10, deduplicate_pages=True)
    res = Retriever(CHUNKS, cfg).search("offside", top_k=3)
    scores = [x.score for x in res]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_no_results():
    r = Retriever(CHUNKS, replace(DEFAULT_CONFIG, candidate_pool=10))
    assert r.search("the and of") == []  # all stopwords


def test_validate_rejects_missing_keys():
    with pytest.raises(ChunkValidationError):
        validate_chunks([{"id": 0, "page": 1}])


def test_validate_rejects_bad_page():
    with pytest.raises(ChunkValidationError):
        validate_chunks([{"id": 0, "page": 0, "text": "x"}])
    with pytest.raises(ChunkValidationError):
        validate_chunks([{"id": 0, "page": "1", "text": "x"}])


def test_validate_rejects_duplicate_ids():
    with pytest.raises(ChunkValidationError):
        validate_chunks([
            {"id": 0, "page": 1, "text": "a"},
            {"id": 0, "page": 2, "text": "b"},
        ])


def test_validate_rejects_empty_text():
    with pytest.raises(ChunkValidationError):
        validate_chunks([{"id": 0, "page": 1, "text": "   "}])


def test_diagnostics_fields():
    r = Retriever(CHUNKS, replace(DEFAULT_CONFIG, candidate_pool=10))
    res = r.search("offside")
    d = r.diagnostics("offside", res)
    assert {"num_results", "top_score", "score_gap", "unique_matched_terms"} <= d.keys()
