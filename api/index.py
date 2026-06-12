"""FIFA Laws of the Game RAG chatbot — Vercel Python serverless API.

RAG pipeline:
  1. Retrieve: BM25 over pre-chunked PDF text (data/chunks.json) — pure Python,
     index built once per cold start (<50 ms for ~300 chunks), zero retrieval latency.
  2. Generate: Groq llama-3.1-8b-instant (free tier, ~800 tok/s) grounded on
     the retrieved chunks, with page citations.
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------- retrieval

STOPWORDS = set(
    "a an and are as at be by for from has have if in into is it its of on or "
    "s t that the their there these this to was were what when which who will "
    "with within without you your".split()
)

_token_re = re.compile(r"[a-z0-9']+")


def tokenize(text: str):
    return [t for t in _token_re.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


# Query expansion: colloquial football terms -> the official terminology used in
# the Laws of the Game (taken from the document's own glossary, Law 12, Law 7).
# Motivated by error analysis: users say "red card", the Laws say "sending-off".
SYNONYMS = {
    "red card": "sending-off sent off sending off offences",
    "yellow card": "caution cautionable offences",
    "booked": "caution",
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
    "how long": "duration periods",
    "hand ball": "handball",
}


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [exp for phrase, exp in SYNONYMS.items() if phrase in q]
    return q + " " + " ".join(extra) if extra else q


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in docs_tokens]
        self.avg_len = sum(self.doc_len) / max(len(docs_tokens), 1)
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5, expand: bool = True):
        q = tokenize(expand_query(query) if expand else query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:top_k]


DATA = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
CHUNKS = json.loads(DATA.read_text())
INDEX = BM25([tokenize(c["text"]) for c in CHUNKS])

# ---------------------------------------------------------------- generation

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the FIFA/IFAB \
Laws of the Game 2025/26. Answer ONLY using the provided context excerpts from the \
official document. Cite page numbers like [p. 93] after the facts they support. \
If the context does not contain the answer, say so plainly — do not invent rules. \
Be concise and accurate."""


class ChatRequest(BaseModel):
    message: str
    history: list = []  # [{"role": "user"|"assistant", "content": str}]


app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(CHUNKS), "model": MODEL}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return JSONResponse({"error": "GROQ_API_KEY is not set"}, status_code=500)

    # --- retrieve
    hits = INDEX.search(req.message, top_k=5)
    sources = [
        {"page": CHUNKS[i]["page"], "text": CHUNKS[i]["text"], "score": round(s, 2)}
        for s, i in hits
    ]
    context = "\n\n---\n\n".join(f"[Page {s['page']}]\n{s['text']}" for s in sources)

    # --- generate
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.history[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": str(m["content"])[:2000]})
    messages.append(
        {
            "role": "user",
            "content": f"Context excerpts from the Laws of the Game 2025/26:\n\n{context}\n\nQuestion: {req.message}",
        }
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 700},
            )
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"Could not reach Groq: {e}"}, status_code=502)
    if r.status_code != 200:
        return JSONResponse({"error": f"Groq API error {r.status_code}: {r.text[:300]}"}, status_code=502)

    answer = r.json()["choices"][0]["message"]["content"]
    return {
        "answer": answer,
        "sources": [{"page": s["page"], "preview": s["text"][:160] + "…"} for s in sources],
    }


# Serve the chat UI. Vercel routes all traffic to this app AND excludes the
# public/ dir from the function bundle, so the UI ships as api/ui.html.
from fastapi.responses import HTMLResponse

_UI_CANDIDATES = (
    Path(__file__).resolve().parent / "ui.html",
    Path(__file__).resolve().parent.parent / "public" / "index.html",
)


@app.get("/")
def ui():
    for p in _UI_CANDIDATES:
        if p.exists():
            return HTMLResponse(p.read_text())
    return JSONResponse({"error": "UI file not found in bundle"}, status_code=404)
