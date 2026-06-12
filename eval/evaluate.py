"""Retrieval evaluation for the Laws-of-the-Game RAG system.

Compares three retrievers on a hand-labelled eval set (question -> gold pages):
  - BM25 (production system)
  - TF-IDF cosine similarity (classical baseline)
  - Random retrieval (sanity-check floor)

Metrics: Recall@1 / @3 / @5 (a query is a hit if ANY retrieved chunk lies on a
gold page) and MRR@5 (rank of the first gold-page chunk).

Usage: python eval/evaluate.py
"""
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
from index import BM25, tokenize  # reuse the production implementation

CHUNKS = json.loads((ROOT / "data" / "chunks.json").read_text())
EVAL = json.loads((Path(__file__).parent / "eval_set.json").read_text())
K = 5


# ----------------------------------------------------------------- retrievers

def bm25_retriever(expand=True):
    idx = BM25([tokenize(c["text"]) for c in CHUNKS])
    return lambda q: [i for _, i in idx.search(q, top_k=K, expand=expand)]


def tfidf_retriever():
    docs = [tokenize(c["text"]) for c in CHUNKS]
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(n / f) for t, f in df.items()}

    def vec(tokens):
        tf = Counter(tokens)
        v = {t: (1 + math.log(f)) * idf.get(t, 0.0) for t, f in tf.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    doc_vecs = [vec(d) for d in docs]

    def search(q):
        qv = vec(tokenize(q))
        scores = []
        for i, dv in enumerate(doc_vecs):
            s = sum(w * dv.get(t, 0.0) for t, w in qv.items())
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:K]]

    return search


def random_retriever(seed=42):
    rng = random.Random(seed)
    return lambda q: rng.sample(range(len(CHUNKS)), K)


# -------------------------------------------------------------------- metrics

def evaluate(search, name):
    recall = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    misses = []
    for item in EVAL:
        ids = search(item["question"])
        pages = [CHUNKS[i]["page"] for i in ids]
        gold = set(item["gold_pages"])
        first_hit = next((r for r, p in enumerate(pages, 1) if p in gold), None)
        if first_hit:
            mrr += 1.0 / first_hit
            for k in recall:
                if first_hit <= k:
                    recall[k] += 1
        else:
            misses.append((item["question"], pages, sorted(gold)))
    n = len(EVAL)
    print(f"\n{name}")
    print(f"  Recall@1: {recall[1]/n:.2f}   Recall@3: {recall[3]/n:.2f}   "
          f"Recall@5: {recall[5]/n:.2f}   MRR@5: {mrr/n:.2f}")
    for q, got, gold in misses:
        print(f"  MISS: {q!r}  retrieved pages {got}  gold {gold}")
    return {"name": name, "recall@1": recall[1]/n, "recall@3": recall[3]/n,
            "recall@5": recall[5]/n, "mrr@5": mrr/n}


if __name__ == "__main__":
    print(f"{len(EVAL)} queries over {len(CHUNKS)} chunks, k={K}")
    results = [
        evaluate(bm25_retriever(expand=True), "BM25 + query expansion (production)"),
        evaluate(bm25_retriever(expand=False), "BM25 plain"),
        evaluate(tfidf_retriever(), "TF-IDF cosine (baseline)"),
        evaluate(random_retriever(), "Random (floor)"),
    ]
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
