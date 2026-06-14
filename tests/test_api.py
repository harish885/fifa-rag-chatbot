import json
import warnings

import httpx
import pytest
import respx

warnings.filterwarnings("ignore")

# A key must be present at import so the app builds; requests read it lazily.
import os
os.environ.setdefault("GROQ_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from api.index import GROQ_URL, app  # noqa: E402

client = TestClient(app)


def _groq_json(status, answer, cited):
    content = json.dumps({"status": status, "answer": answer, "cited_pages": cited})
    return {"choices": [{"message": {"content": content}}]}


def test_health_no_secret_leak():
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "groq_key_configured" in body
    # only a boolean is exposed, never the key value
    assert body["groq_key_configured"] in (True, False)
    assert "test-key" not in json.dumps(body)


def test_root_serves_ui():
    r = client.get("/")
    assert r.status_code == 200 and "Laws of the Game" in r.text


def test_empty_message_rejected():
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422


def test_overlong_message_rejected():
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 422


def test_invalid_role_rejected():
    r = client.post("/api/chat", json={"message": "hi", "history": [{"role": "system", "content": "x"}]})
    assert r.status_code == 422


def test_missing_api_key_returns_500(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with respx.mock:
        respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_json("answered", "x [p. 87]", [87])))
        r = client.post("/api/chat", json={"message": "How long is half-time?"})
    assert r.status_code == 500


def test_mocked_groq_success_grounded():
    with respx.mock:
        respx.post(GROQ_URL).mock(return_value=httpx.Response(
            200, json=_groq_json("answered", "Half-time is 15 minutes [p. 87].", [87])))
        r = client.post("/api/chat", json={"message": "How long is half-time?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert body["grounded"] is True and body["cited_pages"] == [87]
    assert [s["page"] for s in body["sources"]] == [87]


def test_refusal_hides_irrelevant_sources():
    with respx.mock:
        respx.post(GROQ_URL).mock(return_value=httpx.Response(
            200, json=_groq_json("out_of_scope", "That isn't covered by the Laws.", [])))
        r = client.post("/api/chat", json={"message": "Who won the 2022 World Cup?"})
    body = r.json()
    assert body["status"] == "out_of_scope"
    assert body["sources"] == [] and body["grounded"] is False


def test_invalid_citation_not_shown():
    # Model cites a page that was never retrieved -> not surfaced, not grounded.
    with respx.mock:
        respx.post(GROQ_URL).mock(return_value=httpx.Response(
            200, json=_groq_json("answered", "See [p. 9999].", [9999])))
        r = client.post("/api/chat", json={"message": "How long is half-time?"})
    body = r.json()
    assert body["sources"] == [] and body["grounded"] is False


def test_malformed_groq_response_502():
    with respx.mock:
        respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json={"unexpected": "shape"}))
        r = client.post("/api/chat", json={"message": "offside?"})
    assert r.status_code == 502


def test_groq_timeout_504():
    with respx.mock:
        respx.post(GROQ_URL).mock(side_effect=httpx.TimeoutException("t"))
        r = client.post("/api/chat", json={"message": "offside?"})
    assert r.status_code == 504


def test_groq_auth_failure_500_no_retry():
    with respx.mock:
        route = respx.post(GROQ_URL).mock(return_value=httpx.Response(401, json={"error": "bad"}))
        r = client.post("/api/chat", json={"message": "offside?"})
    assert r.status_code == 500
    assert route.call_count == 1  # auth failures are not retried


def test_rate_limit_429_retried_then_busy(monkeypatch):
    import api.index as idx
    monkeypatch.setattr(idx, "GROQ_BACKOFF_S", 0)  # no real sleeping in tests
    with respx.mock:
        route = respx.post(GROQ_URL).mock(return_value=httpx.Response(429, json={"error": "rate"}))
        r = client.post("/api/chat", json={"message": "offside?"})
    assert r.status_code == 429
    assert route.call_count == idx.GROQ_MAX_RETRIES  # 429 is retried, not given up immediately


def test_rate_limit_recovers_on_retry(monkeypatch):
    import api.index as idx
    monkeypatch.setattr(idx, "GROQ_BACKOFF_S", 0)
    responses = [httpx.Response(429, json={"error": "rate"}),
                 httpx.Response(200, json=_groq_json("answered", "Offside is in Law 11 [p. 103].", [103]))]
    with respx.mock:
        respx.post(GROQ_URL).mock(side_effect=responses)
        r = client.post("/api/chat", json={"message": "What is the offside rule?"})
    assert r.status_code == 200 and r.json()["status"] == "answered"


def test_security_headers_present():
    r = client.get("/api/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
