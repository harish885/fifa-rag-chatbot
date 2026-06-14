from rag.citations import extract_citations, validate_citations


def test_extract_single():
    assert extract_citations("Half-time is 15 minutes [p. 87].") == [87]


def test_extract_multiple_and_dedup():
    assert extract_citations("[p. 117] and [pp. 118, 117]") == [117, 118]


def test_extract_variants():
    assert extract_citations("[p.93] [page 94] [pp 95]") == [93, 94, 95]


def test_extract_none():
    assert extract_citations("No citation here.") == []
    assert extract_citations("") == []


def test_validate_splits_supported_unsupported():
    supported, unsupported = validate_citations([87, 999], [87, 88])
    assert supported == [87] and unsupported == [999]
