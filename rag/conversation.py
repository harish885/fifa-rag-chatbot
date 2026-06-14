"""History-aware retrieval-query construction (deterministic).

Problem this solves
--------------------
A user asks "What is the offside rule?" then "What are the exceptions?". Naive
retrieval runs BM25 on the bare follow-up ("exceptions") and drifts to unrelated
text (e.g. injury exceptions). We detect referential / underspecified follow-ups
and prepend the *subject of the most recent user turn* so retrieval stays on
topic.

Design choices
--------------
* Deterministic heuristic - no LLM, no API key, so the offline evaluation and CI
  never depend on a network call. (A future LLM rewrite, if added, must keep a
  deterministic fallback and temperature 0.)
* We borrow the subject from the most recent *user* turn only. Assistant text is
  never mixed into the retrieval query, so generated prose cannot dominate it.
* History is bounded (turn count and per-message length) and malformed history
  is skipped, not trusted.
* The result is inspectable (`RetrievalQuery`) for debugging, but the rewritten
  string is not surfaced in the normal UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import DEFAULT_CONFIG, RetrievalConfig
from .text import tokenize

# Pronouns / demonstratives that signal the question refers back to something.
_REFERENTIAL_MARKERS = frozenset(
    "it its that this then those these they them there he she him her his".split()
)

# Head words that are meaningless without an antecedent topic.
_UNDERSPECIFIED_HEADS = frozenset(
    "exception exceptions difference differences rest others one ones "
    "consequence consequences result results".split()
)

# Generic football roles; a short question built only around these is a follow-up.
_GENERIC_ROLES = frozenset("player players goalkeeper goalie keeper team".split())

_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class RetrievalQuery:
    """The query actually sent to the retriever, plus debug provenance."""

    text: str               # query used for retrieval (possibly rewritten)
    original: str           # the user's raw current message
    rewritten: bool         # whether history context was injected
    subject_source: Optional[str] = None  # prior user turn used, if any


def _clean_history(history: Optional[Sequence], turns: int, max_chars: int) -> List[dict]:
    """Return the last ``turns`` well-formed messages, content length-capped.

    Malformed entries (non-dict, bad role, empty/non-string content) are dropped
    silently. Never raises on bad input.
    """
    if not history:
        return []
    cleaned: List[dict] = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content.strip()[:max_chars]})
    return cleaned[-turns * 2:] if turns else []


def _is_referential(message: str) -> bool:
    """Heuristic: does ``message`` depend on prior context to be answerable?"""
    norm = re.sub(r"\s+", " ", message.lower()).strip()
    if not norm:
        return False
    if norm.startswith("what about"):
        return True
    words = _WORD_RE.findall(norm)
    if any(w in _REFERENTIAL_MARKERS for w in words):
        return True
    content = tokenize(norm)
    if not content:
        return True  # e.g. "and?" / pure stopwords -> needs context
    if any(w in _UNDERSPECIFIED_HEADS for w in content):
        return True
    # Short question whose only nouns are generic roles ("the goalkeeper").
    if len(content) <= 4 and any(w in _GENERIC_ROLES for w in content):
        return True
    return False


def _last_user_subject(history: Sequence[dict]) -> Optional[str]:
    """Most recent prior *user* message (the topic to carry forward)."""
    for m in reversed(history):
        if m["role"] == "user":
            return m["content"]
    return None


def build_retrieval_query(
    message: str,
    history: Optional[Sequence] = None,
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> RetrievalQuery:
    """Construct the retrieval query, injecting prior topic for follow-ups.

    Standalone questions are returned unchanged. For referential follow-ups we
    append the distinctive content tokens of the most recent user turn that are
    not already present in the current message, bounded by ``max_query_chars``.
    """
    raw = (message or "").strip()
    if not raw:
        return RetrievalQuery(text="", original="", rewritten=False)

    if not _is_referential(raw):
        return RetrievalQuery(text=raw, original=raw, rewritten=False)

    hist = _clean_history(history, config.history_turns, config.max_query_chars)
    subject = _last_user_subject(hist)
    if not subject:
        return RetrievalQuery(text=raw, original=raw, rewritten=False)

    present = set(tokenize(raw))
    subject_terms = [t for t in tokenize(subject) if t not in present]
    if not subject_terms:
        return RetrievalQuery(text=raw, original=raw, rewritten=False, subject_source=subject)

    rewritten = f"{raw} {' '.join(subject_terms)}"[: config.max_query_chars]
    return RetrievalQuery(text=rewritten, original=raw, rewritten=True, subject_source=subject)
