"""Deterministic evaluation regression check for CI.

Recomputes the retrieval metrics in-memory and compares them to the committed
``eval/results.json`` (ignoring the run timestamp, which legitimately changes).
Exits non-zero if any system's metrics drift, so CI fails on an unintended
retrieval change without the timestamp dirtying the working tree.

Usage: python eval/check_regression.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "eval" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

RESULTS = ROOT / "eval" / "results.json"


def main() -> int:
    if not RESULTS.exists():
        print("results.json missing — run python eval/evaluate.py", file=sys.stderr)
        return 1
    committed = json.loads(RESULTS.read_text(encoding="utf-8"))

    from dataclasses import replace
    from rag import DEFAULT_CONFIG, Retriever, load_chunks

    chunks = load_chunks(ROOT / "data" / "chunks.json")
    dataset = json.loads((ROOT / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    items = dataset["items"]
    cfg = DEFAULT_CONFIG
    builders = {
        "BM25 + expansion + page-dedup (production)": Retriever(chunks, cfg),
        "BM25 + expansion, chunk-level (no dedup)": Retriever(chunks, replace(cfg, deduplicate_pages=False)),
        "BM25 plain + page-dedup (ablation: no expansion)": Retriever(chunks, replace(cfg, expand_query=False)),
        "BM25 plain, chunk-level (no expansion, no dedup)": Retriever(chunks, replace(cfg, expand_query=False, deduplicate_pages=False)),
    }
    fresh = {name: evaluate.evaluate_system(name, evaluate.bm25_search(r), items)["metrics"]
             for name, r in builders.items()}

    drift = []
    for sysrec in committed["systems"]:
        name = sysrec["name"]
        if name not in fresh:
            continue  # TF-IDF / random reuse local baselines; skip exact recompute
        for k, v in fresh[name].items():
            if sysrec["metrics"].get(k) != v:
                drift.append(f"{name}: {k} committed={sysrec['metrics'].get(k)} now={v}")
    if drift:
        print("EVALUATION REGRESSION DETECTED:", file=sys.stderr)
        for d in drift:
            print("  " + d, file=sys.stderr)
        print("Re-run `python eval/evaluate.py` and review/commit the change.", file=sys.stderr)
        return 1
    print("Evaluation regression check passed (metrics match committed results.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
