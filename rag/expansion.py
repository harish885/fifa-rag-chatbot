"""Colloquial -> official query expansion for the Laws of the Game.

Why this exists
---------------
Users ask in everyday football language ("red card", "stoppage time") while the
corpus is written in precise legal terminology ("sending-off offence",
"additional time allowance"). Lexical retrieval (BM25) scores on shared surface
terms, so this register gap costs recall. The map below bridges it.

Provenance of the map (stated honestly)
---------------------------------------
Entries fall into two groups, flagged per row in ``SYNONYM_SOURCES``:

* ``glossary``  - the colloquial->official pairing is attested directly in the
  document's own Glossary / Law text (e.g. "sending-off", "caution",
  "additional time", "kicks from the penalty mark").
* ``dev``       - the pairing was *motivated by error analysis on the 20-query
  development set* (e.g. mapping the bare phrase "how long" toward
  "duration periods"). These are development-derived and therefore the recall
  gains they produce on that set are optimistic, not held-out evidence.

Matching is **token-boundary aware**: "keeper" expands a standalone word but is
NOT triggered inside "goalkeeper", and "var" is not triggered inside unrelated
words. Multi-word phrases are supported with flexible internal whitespace.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# phrase (already lowercase) -> space-separated official expansion terms.
SYNONYMS: Dict[str, str] = {
    "red card": "sending-off sent-off send-off offences",
    "yellow card": "caution cautionable offences",
    "booked": "caution cautioned",
    "booking": "caution",
    "shoot-out": "kicks from the penalty mark",
    "shootout": "kicks from the penalty mark",
    "penalty shoot": "kicks from the penalty mark",
    "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "pitch": "field of play",
    "var": "video assistant referee",
    "injury time": "additional time allowance",
    "stoppage time": "additional time allowance",
    "added time": "additional time allowance",
    "how long": "duration period",
    "hand ball": "handball",
}

# Provenance tag per entry: "glossary" (traceable to the document) or
# "dev" (motivated by development-set error analysis).
SYNONYM_SOURCES: Dict[str, str] = {
    "red card": "glossary",
    "yellow card": "glossary",
    "booked": "glossary",
    "booking": "glossary",
    "shoot-out": "glossary",
    "shootout": "glossary",
    "penalty shoot": "glossary",
    "goalie": "glossary",
    "keeper": "glossary",
    "pitch": "glossary",
    "var": "glossary",
    "injury time": "glossary",
    "stoppage time": "glossary",
    "added time": "glossary",
    "how long": "dev",
    "hand ball": "glossary",
}


def _phrase_pattern(phrase: str) -> "re.Pattern[str]":
    """Compile a word-boundary regex for ``phrase`` allowing flexible whitespace.

    Internal spaces become ``\\s+`` so "hand   ball" still matches "hand ball".
    Boundaries (``\\b``) prevent matching inside larger words.
    """
    parts = [re.escape(p) for p in phrase.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"\b{body}\b")


# Pre-compile once; longest phrases first so overlapping maps are checked in a
# stable, specific-to-general order.
_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (phrase, _phrase_pattern(phrase))
    for phrase in sorted(SYNONYMS, key=lambda p: (-len(p), p))
]


def _normalize(query: str) -> str:
    """Lowercase and collapse runs of whitespace to single spaces."""
    return re.sub(r"\s+", " ", query.lower()).strip()


def matched_phrases(query: str) -> List[str]:
    """Return the synonym-map phrases that fire for ``query`` (deterministic order)."""
    norm = _normalize(query)
    if not norm:
        return []
    return [phrase for phrase, pat in _PATTERNS if pat.search(norm)]


def expand_query(query: str) -> str:
    """Append official-terminology expansions for any colloquial phrase present.

    Returns the normalized query followed by deduplicated expansion terms. The
    original query terms are preserved; expansion only *adds* signal. Output is
    deterministic for a given input.
    """
    norm = _normalize(query)
    if not norm:
        return ""
    present = set(norm.split())
    extra: List[str] = []
    for phrase in matched_phrases(norm):
        for term in SYNONYMS[phrase].split():
            if term not in present and term not in extra:
                extra.append(term)
    return norm if not extra else f"{norm} {' '.join(extra)}"
