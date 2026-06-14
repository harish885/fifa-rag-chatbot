"""Retrieval evaluation for the Laws-of-the-Game RAG system.

Metric (named honestly)
-----------------------
We report **Page Hit Rate@k** (``hit@k`` in code/JSON), NOT information-retrieval
recall. Definition:

    A query is a *hit at k* when at least one of the top-k retrieved chunks comes
    from a page labelled as containing the answer.

Limitations (carried in the output and the docs):
  * The correct page does not guarantee the chunk contains the answer span.
  * Page labels are coarse; a labelled page may hold other material too.
  * A query gets a single binary success value; this does not measure how many
    relevant chunks were retrieved.
We also report page-level MRR@5 (rank of the first gold-page chunk) and the mean
number of unique pages in the top-5.

This harness imports the SAME retrieval core as the production API (``rag/``),
so the two cannot diverge. It needs no API key and runs fully offline.

Usage: python eval/evaluate.py [--output eval/results.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))
from rag import DEFAULT_CONFIG, Retriever, load_chunks, matched_phrases, tokenize  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "chunks.json"
EVAL_PATH = Path(__file__).parent / "eval_set.json"
KS = (1, 3, 5)
K = 5


# ------------------------------------------------------------------- utilities

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> List[float]:
    """95% Wilson score interval for a binomial proportion (rounded)."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3)]


# ------------------------------------------------------------------ retrievers
# Each retriever returns the ordered list of (page, expansion_phrases) for a
# query. BM25 variants reuse the production Retriever with config overrides;
# TF-IDF and random are local baselines.

def bm25_search(retriever: Retriever) -> Callable[[str], List[int]]:
    def search(q: str) -> List[int]:
        return [r.page for r in retriever.search(q, top_k=K)]
    return search


def tfidf_search(chunks: List[dict]) -> Callable[[str], List[int]]:
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(n / f) for t, f in df.items()}

    def vec(tokens):
        tf = Counter(tokens)
        v = {t: (1 + math.log(f)) * idf.get(t, 0.0) for t, f in tf.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    doc_vecs = [vec(d) for d in docs]

    def search(q: str) -> List[int]:
        qv = vec(tokenize(q))
        scored = []
        for i, dv in enumerate(doc_vecs):
            s = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda si: (-si[0], si[1]))
        return [chunks[i]["page"] for _, i in scored[:K]]

    return search


def random_search(chunks: List[dict], seed: int = 42) -> Callable[[str], List[int]]:
    rng = random.Random(seed)
    pages = [c["page"] for c in chunks]
    return lambda q: rng.sample(pages, K)


# --------------------------------------------------------------------- scoring

def evaluate_system(name: str, search: Callable[[str], List[int]], items: List[dict]) -> Dict:
    hit_counts = {k: 0 for k in KS}
    mrr = 0.0
    unique_pages_sum = 0
    per_query: List[Dict] = []
    misses: List[Dict] = []
    for item in items:
        q = item["question"]
        pages = search(q)
        gold = set(item["gold_pages"])
        first_hit = next((rank for rank, p in enumerate(pages, 1) if p in gold), None)
        unique_pages_sum += len(set(pages))
        if first_hit:
            mrr += 1.0 / first_hit
            for k in KS:
                if first_hit <= k:
                    hit_counts[k] += 1
        else:
            misses.append({"question": q, "retrieved_pages": pages, "gold_pages": sorted(gold)})
        per_query.append({
            "question": q,
            "gold_pages": sorted(gold),
            "retrieved_pages": pages,
            "unique_pages": len(set(pages)),
            "first_hit_rank": first_hit,
            "hit@5": first_hit is not None and first_hit <= 5,
            "expansion_phrases": matched_phrases(q),
        })
    n = len(items)
    metrics = {f"hit@{k}": round(hit_counts[k] / n, 3) for k in KS}
    metrics.update({
        "page_mrr@5": round(mrr / n, 3),
        "mean_unique_pages@5": round(unique_pages_sum / n, 2),
        "hit@5_count": f"{hit_counts[5]}/{n}",
        "hit@5_wilson_ci95": wilson_interval(hit_counts[5], n),
    })
    return {"name": name, "metrics": metrics, "per_query": per_query, "misses": misses}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (Page Hit Rate@k).")
    parser.add_argument("--output", "-o", default=str(Path(__file__).parent / "results.json"))
    args = parser.parse_args(argv)

    chunks = load_chunks(CHUNKS_PATH)
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    items = dataset["items"] if isinstance(dataset, dict) else dataset

    cfg = DEFAULT_CONFIG
    prod = Retriever(chunks, cfg)                                  # expand + dedup
    expand_nodedup = Retriever(chunks, replace(cfg, deduplicate_pages=False))
    plain = Retriever(chunks, replace(cfg, expand_query=False))    # no expansion, dedup
    plain_nodedup = Retriever(chunks, replace(cfg, expand_query=False, deduplicate_pages=False))

    systems = [
        evaluate_system("BM25 + expansion + page-dedup (production)", bm25_search(prod), items),
        evaluate_system("BM25 + expansion, chunk-level (no dedup)", bm25_search(expand_nodedup), items),
        evaluate_system("BM25 plain + page-dedup (ablation: no expansion)", bm25_search(plain), items),
        evaluate_system("BM25 plain, chunk-level (no expansion, no dedup)", bm25_search(plain_nodedup), items),
        evaluate_system("TF-IDF cosine (baseline)", tfidf_search(chunks), items),
        evaluate_system("Random (floor)", random_search(chunks), items),
    ]

    out = {
        "run": {
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metric": "Page Hit Rate@k (hit@k): a query is a hit at k if any top-k chunk is on a gold page.",
            "dataset": {"name": dataset.get("name"), "role": dataset.get("role"), "n": len(items)},
            "corpus": {"chunks_file": "data/chunks.json", "chunk_count": len(chunks),
                       "chunks_sha256": sha256_file(CHUNKS_PATH)},
            "config": {"top_k": cfg.top_k, "candidate_pool": cfg.candidate_pool,
                       "bm25_k1": cfg.bm25_k1, "bm25_b": cfg.bm25_b},
            "caveat": ("Development set; gains are optimistic, not held-out. Page labels are "
                       "coarse and do not establish answer correctness."),
        },
        "systems": systems,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")

    # Console summary
    print(f"{len(items)} queries over {len(chunks)} chunks (dataset={dataset.get('name')}, role={dataset.get('role')})")
    for s in systems:
        m = s["metrics"]
        print(f"\n{s['name']}")
        print(f"  hit@1 {m['hit@1']:.2f}  hit@3 {m['hit@3']:.2f}  hit@5 {m['hit@5']:.2f} "
              f"({m['hit@5_count']}, 95% CI {m['hit@5_wilson_ci95']})  "
              f"page-MRR@5 {m['page_mrr@5']:.2f}  uniq-pages@5 {m['mean_unique_pages@5']}")
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
