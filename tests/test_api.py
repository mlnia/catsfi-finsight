import pytest
from fastapi.testclient import TestClient

from finsight.api.app import create_app
from finsight.config import Settings
from finsight.container import build_container

from .conftest import FakeSEC, synthetic_closes


def make_client(tmp_path, **overrides):
    s = Settings(env="test", data_dir=tmp_path, chunk_tokens=120, chunk_overlap_tokens=20, **overrides)
    c = build_container(s, sec=FakeSEC(), fetch_closes=lambda symbol, interval="1d": synthetic_closes())
    return TestClient(create_app(container=c))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path, rate_limit_per_minute=1000) as cl:
        yield cl


def test_health_ui_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"
    assert "CATsFi" in client.get("/").text
    client.get("/v1/documents")
    m = client.get("/metrics").text
    assert "finsight_request_seconds_bucket" in m


def test_ingest_query_flow(client):
    r = client.post("/v1/ingest", json={"tickers": ["NWRB"]})
    assert r.status_code == 200 and r.json()["chunks"] > 5
    docs = client.get("/v1/documents").json()
    assert docs[0]["ticker"] == "NWRB"
    q = client.post("/v1/query", json={"question": "Which supplier provides the lidar sensors?", "filters": {"ticker": "NWRB"}})
    body = q.json()
    assert q.status_code == 200 and "Kestrel" in body["answer"] and body["sources"]
    assert "X-Response-Time-ms" in q.headers
    dense = client.post("/v1/search", json={"query": "gross margin", "mode": "dense", "top_k": 3}).json()
    assert len(dense["results"]) == 3
    sparse = client.post("/v1/search", json={"query": "gross margin", "mode": "sparse", "top_k": 3}).json()
    assert sparse["results"] and all("margin" in r["text"].lower() for r in sparse["results"])  # lexical match only
    assert client.post("/v1/query", json={"question": "abc", "prompt_version": "rag_answer@v9"}).status_code == 400


def test_agent_approval_over_http(client):
    r = client.post("/v1/agent/chat", json={"message": "Ingest the latest 10-K filing for NWRB"}).json()
    assert r["status"] == "awaiting_approval"
    tid = r["thread_id"]
    assert client.post("/v1/agent/chat", json={"message": "hello", "thread_id": tid}).status_code == 409
    assert client.get("/v1/agent/pending").json()[0]["thread_id"] == tid
    done = client.post(f"/v1/agent/{tid}/approval", json={"approved": True}).json()
    assert done["status"] == "completed"
    assert client.post(f"/v1/agent/{tid}/approval", json={"approved": True}).status_code == 404


def test_validation_errors(client):
    assert client.post("/v1/query", json={"question": "x"}).status_code == 422
    assert client.post("/v1/agent/chat", json={"message": "hi there", "thread_id": "../etc"}).status_code == 422


def test_auth_and_roles(tmp_path):
    with make_client(tmp_path, api_keys="viewer-key:viewer,admin-key:admin") as cl:
        assert cl.post("/v1/query", json={"question": "revenue?"}).status_code == 401
        assert cl.post("/v1/query", json={"question": "revenue?"}, headers={"X-API-Key": "nope"}).status_code == 401
        assert cl.post("/v1/query", json={"question": "revenue?"}, headers={"X-API-Key": "viewer-key"}).status_code == 200
        assert cl.post("/v1/ingest", json={"tickers": ["NWRB"]}, headers={"X-API-Key": "viewer-key"}).status_code == 403
        assert cl.post("/v1/ingest", json={"tickers": ["NWRB"]}, headers={"X-API-Key": "admin-key"}).status_code == 200


def test_rate_limit(tmp_path):
    with make_client(tmp_path, rate_limit_per_minute=3) as cl:
        codes = [cl.get("/v1/documents").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[-1] == 429
