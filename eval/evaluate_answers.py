"""Answer-level evaluation harness (human-first, with deterministic checks).

This complements retrieval evaluation. Retrieval hit rate does NOT establish
answer correctness, so this harness:

  * runs the SAME history-aware retrieval as production for each item,
  * (optionally, with a key) generates an answer with temperature 0,
  * runs deterministic citation checks (cited pages must have been retrieved),
  * emits an annotation template for HUMAN rubric scoring (the authoritative
    signal — see eval/README.md).

Modes
-----
  python eval/evaluate_answers.py                 # retrieval + annotation template (no key)
  python eval/evaluate_answers.py --generate      # also generate answers (needs GROQ_API_KEY)
  python eval/evaluate_answers.py --citation-check-only   # re-check citations in answer_results.json

Human scores are never invented. If no answers are present, items are marked
"pending_generation"; if answers exist but no human has scored them, the rubric
fields stay null and the item is "pending_human_review".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag import (  # noqa: E402
    DEFAULT_CONFIG,
    Retriever,
    build_retrieval_query,
    extract_citations,
    validate_citations,
)

SET_PATH = Path(__file__).parent / "answer_eval_set.json"
RESULTS_PATH = Path(__file__).parent / "answer_results.json"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Kept in sync conceptually with api/index.py SYSTEM_PROMPT (JSON envelope).
SYSTEM_PROMPT = (
    "You answer questions about the IFAB/FIFA Laws of the Game 2025/26 using ONLY "
    "the provided context. Reply with a single JSON object: "
    '{"status": "answered"|"insufficient_context"|"out_of_scope", "answer": string, '
    '"cited_pages": [int]}. Cite pages inline as [p. N]. Never invent rules or pages.'
)

RUBRIC_FIELDS = {
    "correctness": None,            # 0/1/2
    "faithfulness": None,           # 0/1/2
    "citation_correctness": None,   # 0/1/2
    "citation_completeness": None,  # 0/1/2
    "refusal_behavior": None,       # pass/fail/not-applicable
    "notes": "",                    # human explanation
}


def _generate(messages: list) -> str:
    import httpx

    api_key = os.environ["GROQ_API_KEY"]
    payload = {"model": MODEL, "messages": messages, "temperature": 0,
               "max_tokens": 700, "response_format": {"type": "json_object"}}
    with httpx.Client(timeout=30, trust_env=False) as client:
        r = client.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _answer_from_content(content: str) -> dict:
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "answer" in obj:
            return {"status": obj.get("status", "answered"), "answer": str(obj["answer"])}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {"status": "answered", "answer": content}


def citation_check(answer: str, status: str, retrieved_pages: List[int]) -> dict:
    """Deterministic, key-free citation diagnostics for one answer."""
    cited = extract_citations(answer)
    supported, unsupported = validate_citations(cited, retrieved_pages)
    is_refusal = status in ("insufficient_context", "out_of_scope")
    substantive_no_citation = (not is_refusal) and (not cited)
    return {
        "cited_pages": cited,
        "supported_citations": supported,
        "unsupported_citations": unsupported,  # cited a page that was NOT retrieved -> red flag
        "is_refusal": is_refusal,
        "substantive_answer_without_citation": substantive_no_citation,
    }


def build_items(generate: bool) -> dict:
    dataset = json.loads(SET_PATH.read_text(encoding="utf-8"))
    retriever = Retriever(config=DEFAULT_CONFIG)
    out_items = []
    for item in dataset["items"]:
        history = item.get("history", [])
        rq = build_retrieval_query(item["question"], history, DEFAULT_CONFIG)
        results = retriever.search(rq.text)
        retrieved_pages = [r.page for r in results]
        record = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "rewritten_retrieval_query": rq.text if rq.rewritten else None,
            "retrieved_pages": retrieved_pages,
            "sources": [{"page": r.page, "preview": r.preview()} for r in results],
            "model": MODEL,
            "answer": None,
            "answer_status": None,
            "citation_check": None,
            "human_scores": dict(RUBRIC_FIELDS),
            "state": "pending_generation",
        }
        if generate:
            context = "\n\n---\n\n".join(f"[Page {r.page}]\n{r.text}" for r in results)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += [{"role": m["role"], "content": m["content"]} for m in history]
            messages.append({"role": "user",
                             "content": f"Context:\n\n{context}\n\nQuestion: {item['question']}"})
            parsed = _answer_from_content(_generate(messages))
            record["answer"] = parsed["answer"]
            record["answer_status"] = parsed["status"]
            record["citation_check"] = citation_check(parsed["answer"], parsed["status"], retrieved_pages)
            record["state"] = "pending_human_review"
        out_items.append(record)
    return {
        "run": {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": dataset.get("name"),
            "model": MODEL if generate else None,
            "generated": generate,
            "scoring": "HUMAN rubric is authoritative; see eval/README.md. Automated citation checks are diagnostic only.",
        },
        "items": out_items,
    }


def recheck_citations() -> dict:
    if not RESULTS_PATH.exists():
        raise SystemExit("No answer_results.json — run with --generate first.")
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    n = 0
    for rec in data.get("items", []):
        if rec.get("answer"):
            rec["citation_check"] = citation_check(
                rec["answer"], rec.get("answer_status") or "answered", rec["retrieved_pages"])
            n += 1
    print(f"Re-checked citations for {n} answered items.")
    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Answer-level evaluation harness.")
    parser.add_argument("--generate", action="store_true", help="generate answers (needs GROQ_API_KEY)")
    parser.add_argument("--citation-check-only", action="store_true",
                        help="re-run deterministic citation checks on existing answer_results.json")
    args = parser.parse_args(argv)

    if args.citation_check_only:
        data = recheck_citations()
    else:
        if args.generate and not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("--generate needs GROQ_API_KEY in the environment.")
        data = build_items(generate=args.generate)

    RESULTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    states = {}
    for rec in data["items"]:
        states[rec["state"]] = states.get(rec["state"], 0) + 1
    print(f"Wrote {len(data['items'])} items -> {RESULTS_PATH}  states={states}")
    if not data["run"].get("generated"):
        print("Note: answers NOT generated (no --generate / no key). Human scores pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
