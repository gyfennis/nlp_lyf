import io
import os

import pytest

from app.services.queue import InMemoryParseQueue
from app.services.rag import RagService
from app.services.chat import StubChatModel
from app.services.retrieval import InMemoryVectorStore, StubEmbeddingModel


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_document(client, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("USE_INMEMORY_QUEUE", "1")

    # Rebuild app with new env — use fresh import
    from fastapi.testclient import TestClient
    from importlib import reload
    import app.config as config_mod
    import app.main as main_mod

    config_mod.get_settings.cache_clear()
    reload(main_mod)
    with TestClient(main_mod.app) as c:
        pdf = io.BytesIO(b"%PDF-1.4 fake content")
        r = c.post(
            "/upload/document",
            data={"knowledge_base_id": "kb1"},
            files={"file": ("demo.pdf", pdf, "application/pdf")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "demo.pdf"
    assert body["knowledge_base_id"] == "kb1"
    assert body["filestate"] == "已上传"
    assert os.path.exists(body["filepath"])


def test_chat_stub(client):
    r = client.post(
        "/chat",
        json={"question": "Q3 销售额如何？", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["file_name"] == "sample.pdf"


def test_rag_service_unit():
    hits = [
        {
            "text": "chunk text",
            "db_id": 2,
            "file_name": "x.pdf",
            "file_path": "uploads/x.pdf",
        }
    ]
    svc = RagService(
        StubEmbeddingModel(),
        InMemoryVectorStore(hits),
        StubChatModel(),
    )
    resp = svc.chat("测试问题", top_k=1)
    assert "测试问题" in resp.answer
    assert resp.sources[0].db_id == 2


def test_inmemory_queue():
    q = InMemoryParseQueue()
    q.enqueue_parse_job(file_id=1, file_name="a.pdf", file_path="/tmp/a.pdf")
    assert q.jobs[0]["id"] == 1
