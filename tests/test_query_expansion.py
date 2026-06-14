from rag.expansion import expand_query, matched_phrases


def test_intended_phrase_match():
    assert "sending-off" in expand_query("what happens after a red card")
    assert "red card" in matched_phrases("what happens after a red card")


def test_word_boundary_no_substring_keeper_in_goalkeeper():
    # "keeper" must NOT fire inside "goalkeeper".
    assert "keeper" not in matched_phrases("where does the goalkeeper stand")


def test_keeper_standalone_does_fire():
    assert "keeper" in matched_phrases("where does the keeper stand")


def test_no_accidental_var_substring():
    # "var" must not match inside "variation"/"every".
    assert matched_phrases("every variation of the rule") == []


def test_var_standalone_fires():
    assert "var" in matched_phrases("what is var")


def test_no_duplicate_expansion_terms():
    # Two phrases mapping to the same expansion must not append it twice.
    out = expand_query("injury time and stoppage time")  # both -> additional time allowance
    assert out.split().count("additional") == 1
    assert out.split().count("allowance") == 1


def test_multiword_phrase_flexible_whitespace():
    assert "handball" in expand_query("is hand   ball an offence")


def test_capitalization_and_punctuation():
    assert "caution" in expand_query("Was he BOOKED?")


def test_empty_query():
    assert expand_query("") == ""
    assert matched_phrases("") == []


def test_deterministic():
    q = "red card and yellow card and var"
    assert expand_query(q) == expand_query(q)
