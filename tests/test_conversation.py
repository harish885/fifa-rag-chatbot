from rag.conversation import build_retrieval_query


def _hist(*pairs):
    out = []
    for u, a in pairs:
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": a})
    return out


def test_standalone_question_unchanged():
    q = build_retrieval_query("What is the offside rule?", [])
    assert not q.rewritten and q.text == "What is the offside rule?"


def test_standalone_with_history_still_unchanged():
    # A self-contained question should ignore prior history.
    h = _hist(("How long is half-time?", "15 minutes."))
    q = build_retrieval_query("When is a handball an offence?", h)
    assert not q.rewritten


def test_offside_exceptions_followup_gets_topic():
    h = _hist(("What is the offside rule?", "Offside is defined in Law 11."))
    q = build_retrieval_query("What are the exceptions?", h)
    assert q.rewritten and "offside" in q.text.lower()


def test_penalty_goalkeeper_followup():
    h = _hist(("How is a penalty kick taken?", "The ball is placed on the penalty mark."))
    q = build_retrieval_query("Where must the goalkeeper stand?", h)
    assert q.rewritten and "penalty" in q.text.lower()


def test_red_card_replacement_followup():
    h = _hist(("What happens after a red card?", "The player leaves the field."))
    q = build_retrieval_query("Can the player be replaced?", h)
    assert q.rewritten and ("red" in q.text.lower() or "card" in q.text.lower())


def test_empty_history_followup_left_alone():
    q = build_retrieval_query("What are the exceptions?", [])
    assert not q.rewritten


def test_malformed_history_ignored():
    bad = ["not a dict", {"role": "system", "content": "x"}, {"role": "user"}, 42]
    q = build_retrieval_query("What are the exceptions?", bad)
    assert not q.rewritten  # nothing usable -> no crash, no rewrite


def test_excessive_history_is_bounded():
    h = _hist(*[(f"q{i} about throw-in", f"a{i}") for i in range(50)])
    h += _hist(("What is the offside rule?", "Offside is in Law 11."))
    q = build_retrieval_query("What are the exceptions?", h)
    assert q.rewritten and "offside" in q.text.lower()
    assert len(q.text) <= 400


def test_only_user_turns_used_not_assistant():
    h = [{"role": "user", "content": "What is the offside rule?"},
         {"role": "assistant", "content": "penalty corner throwin unrelated words"}]
    q = build_retrieval_query("What are the exceptions?", h)
    assert "offside" in q.text.lower()
    assert "throwin" not in q.text.lower()
