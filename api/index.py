"""FIFA Laws of the Game RAG chatbot — Vercel Python serverless API.

Pipeline (retrieval lives in the reusable ``rag`` package, shared with eval):
  1. Build a history-aware retrieval query (deterministic).
  2. Retrieve page-diverse top-k chunks with BM25 + query expansion.
  3. Ask Groq to answer ONLY from the retrieved context, returning structured
     JSON (status + answer + cited pages).
  4. Validate the model output; surface a clear status and show source badges
     ONLY for pages the answer actually cites.

No secrets are logged. Raw upstream bodies are never returned to clients.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Make the repo root importable so `rag` resolves under Vercel and locally.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import (  # noqa: E402
    DEFAULT_CONFIG,
    Retriever,
    build_retrieval_query,
    extract_citations,
    validate_citations,
)

logger = logging.getLogger("laws_rag")

# --------------------------------------------------------------- limits / config
MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CONTENT_CHARS = 2000
CONTEXT_TOP_K = 8  # chunks (distinct pages) handed to the generator
GROQ_TIMEOUT_S = 25.0
GROQ_MAX_RETRIES = 2  # transient errors only

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = (
    "You answer questions about the IFAB/FIFA Laws of the Game 2025/26 using ONLY "
    "the provided context excerpts from the official document. Always reply with a "
    "single JSON object and nothing else:\n"
    '{"status": "answered" | "insufficient_context" | "out_of_scope", '
    '"answer": string, "cited_pages": [int, ...]}\n'
    "- status \"answered\": the context contains the answer. Write a COMPLETE, "
    "self-contained answer of 1-3 full sentences (never a sentence fragment). "
    "Restate what is being asked and include every specific figure, measurement "
    "and unit found in the context (e.g. distances in metres, times in minutes). "
    "If the question asks for dimensions, give each dimension with its number and "
    "unit. Cite the page(s) you used inline as [p. N] and in cited_pages.\n"
    "- status \"insufficient_context\": the question is about the Laws but the "
    "context does not contain the answer. Say so plainly; cited_pages = [].\n"
    "- status \"out_of_scope\": the question is not about the Laws of the Game "
    "(e.g. match results, history, players). Decline briefly; cited_pages = [].\n"
    "Never invent rules, figures or page numbers; use only what the context states."
)

# Build the retriever once per cold start.
RETRIEVER = Retriever(config=DEFAULT_CONFIG)


# ----------------------------------------------------------------- API models
Role = Literal["user", "assistant"]


class HistoryMessage(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _cap_content(cls, v: str) -> str:
        return v[:MAX_HISTORY_CONTENT_CHARS]


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryMessage] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message too long (max {MAX_MESSAGE_CHARS} chars)")
        return v

    @field_validator("history")
    @classmethod
    def _cap_history(cls, v: List[HistoryMessage]) -> List[HistoryMessage]:
        return v[-MAX_HISTORY_MESSAGES:]


class Source(BaseModel):
    page: int
    preview: str
    score: float


class ChatResponse(BaseModel):
    status: Literal["answered", "insufficient_context", "out_of_scope", "upstream_error"]
    answer: str
    grounded: bool
    cited_pages: List[int] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    request_id: str


# ------------------------------------------------------------ upstream errors
class ConfigError(Exception):
    ...


class UpstreamTimeout(Exception):
    ...


class UpstreamHTTPError(Exception):
    ...


class UpstreamMalformed(Exception):
    ...


# ----------------------------------------------------------- pure helpers (tested)
def parse_model_output(content: str) -> dict:
    """Parse the model's JSON envelope, with a deterministic fallback.

    Returns a dict with keys status/answer/cited_pages. If the content is not the
    expected JSON, fall back to treating it as a plain answer and inferring the
    status from whether it contains page citations.
    """
    content = (content or "").strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "answer" in obj:
            status = obj.get("status")
            if status not in ("answered", "insufficient_context", "out_of_scope"):
                status = "answered" if extract_citations(str(obj["answer"])) else "insufficient_context"
            cited = obj.get("cited_pages") or extract_citations(str(obj["answer"]))
            cited = [int(p) for p in cited if isinstance(p, (int, str)) and str(p).isdigit()]
            return {"status": status, "answer": str(obj["answer"]).strip(), "cited_pages": sorted(set(cited))}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: raw text answer.
    cited = extract_citations(content)
    status = "answered" if cited else "insufficient_context"
    return {"status": status, "answer": content, "cited_pages": cited}


def finalize_response(parsed: dict, results, request_id: str) -> ChatResponse:
    """Apply grounding rules: show source badges only for genuinely cited pages."""
    retrieved_pages = [r.page for r in results]
    status = parsed["status"]
    answer = parsed["answer"] or "I couldn't find that in the Laws of the Game."

    if status != "answered":
        # Refusal / insufficient: no source badges, nothing implied as verified.
        return ChatResponse(status=status, answer=answer, grounded=False,
                            cited_pages=[], sources=[], request_id=request_id)

    supported, _unsupported = validate_citations(parsed["cited_pages"], retrieved_pages)
    by_page = {}
    for r in results:
        by_page.setdefault(r.page, r)
    sources = [
        Source(page=p, preview=by_page[p].preview(), score=round(by_page[p].score, 2))
        for p in supported if p in by_page
    ]
    # Grounded only when the answer cites at least one retrieved page.
    grounded = bool(sources)
    return ChatResponse(status="answered", answer=answer, grounded=grounded,
                        cited_pages=supported, sources=sources, request_id=request_id)


# --------------------------------------------------------------------- app
app = FastAPI(title="Laws of the Game RAG", version="2.0", docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/api/health")
def health():
    """Liveness + corpus info. Never exposes secrets (only whether a key is set)."""
    return {
        "ok": True,
        "chunks": len(RETRIEVER.chunks),
        "model": MODEL,
        "groq_key_configured": bool(os.environ.get("GROQ_API_KEY")),
    }


async def _call_groq(messages: list) -> str:
    """POST to Groq with bounded timeout and limited retries on transient errors.

    Raises typed exceptions; never leaks the API key or raw bodies upward.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ConfigError("GROQ_API_KEY is not set")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            # trust_env=False: ignore ambient proxy/.netrc — talk to Groq directly.
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_S, trust_env=False) as client:
                r = await client.post(
                    GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload
                )
        except httpx.TimeoutException as e:
            last_exc = UpstreamTimeout(str(e))
        except httpx.HTTPError as e:
            last_exc = UpstreamHTTPError(type(e).__name__)
        else:
            if r.status_code == 200:
                try:
                    return r.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError) as e:
                    raise UpstreamMalformed(type(e).__name__)
            if r.status_code in (401, 403):
                raise ConfigError(f"Groq auth failed ({r.status_code})")
            if 400 <= r.status_code < 500:
                raise UpstreamHTTPError(f"client error {r.status_code}")
            last_exc = UpstreamHTTPError(f"server error {r.status_code}")  # retry 5xx
        logger.warning("groq attempt %d failed: %s", attempt, type(last_exc).__name__)
    raise last_exc or UpstreamHTTPError("unknown upstream error")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    request_id = uuid.uuid4().hex[:12]

    # --- retrieve (history-aware) ---
    history = [m.model_dump() for m in req.history]
    rq = build_retrieval_query(req.message, history, DEFAULT_CONFIG)
    results = RETRIEVER.search(rq.text, top_k=CONTEXT_TOP_K)
    diag = RETRIEVER.diagnostics(rq.text, results)
    logger.info("req=%s results=%d top=%.2f rewritten=%s", request_id,
                diag["num_results"], diag["top_score"], rq.rewritten)

    if not results:
        return ChatResponse(
            status="insufficient_context", grounded=False, cited_pages=[], sources=[],
            answer="I couldn't find anything relevant in the Laws of the Game for that question.",
            request_id=request_id,
        )

    context = "\n\n---\n\n".join(f"[Page {r.page}]\n{r.text}" for r in results)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": str(m["content"])[:MAX_HISTORY_CONTENT_CHARS]})
    messages.append({
        "role": "user",
        "content": f"Context excerpts from the Laws of the Game 2025/26:\n\n{context}\n\nQuestion: {req.message}",
    })

    # --- generate ---
    try:
        content = await _call_groq(messages)
    except ConfigError as e:
        return JSONResponse({"error": str(e), "request_id": request_id}, status_code=500)
    except UpstreamTimeout:
        return JSONResponse(
            {"error": "The rules service timed out. Please try again.", "request_id": request_id},
            status_code=504)
    except (UpstreamHTTPError, UpstreamMalformed):
        return JSONResponse(
            {"error": "The rules service is temporarily unavailable.", "request_id": request_id},
            status_code=502)

    parsed = parse_model_output(content)
    return finalize_response(parsed, results, request_id)


# --------------------------------------------------------------------- UI
_UI_CANDIDATES = (
    Path(__file__).resolve().parent / "ui.html",
    ROOT / "public" / "index.html",
)


@app.get("/")
def ui():
    for p in _UI_CANDIDATES:
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
    return JSONResponse({"error": "UI file not found in bundle"}, status_code=404)
