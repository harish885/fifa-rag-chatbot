from rag.text import STOPWORDS, tokenize


def test_lowercases_and_splits():
    assert tokenize("Offside POSITION rule") == ["offside", "position", "rule"]


def test_punctuation_stripped():
    assert tokenize("hand-ball, offence!") == ["hand", "ball", "offence"]


def test_apostrophes_kept_within_token():
    assert "player's" in tokenize("the player's equipment")


def test_stopwords_removed():
    toks = tokenize("what is the offside rule")
    assert "the" not in toks and "is" not in toks
    assert "the" in STOPWORDS


def test_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_numeric_terms_preserved():
    assert "15" in tokenize("the half is 15 minutes")


def test_min_length_drops_single_chars():
    # 's' left from a contraction split is below min_len.
    assert tokenize("a b cd", min_len=2) == ["cd"]
