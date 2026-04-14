import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.server import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_static_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


import io


def test_analyze_missing_prompt(client):
    csv_bytes = b"col_a,col_b\n1,2\n3,4\n"
    resp = client.post(
        "/analyze",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp.status_code == 422


def test_analyze_non_csv_rejected(client):
    resp = client.post(
        "/analyze",
        files={"file": ("data.xlsx", io.BytesIO(b"fake"), "application/octet-stream")},
        data={"prompt": "test question"},
    )
    assert resp.status_code == 400
    assert "CSV" in resp.json()["detail"]


def test_analyze_valid_returns_task_id(client, monkeypatch):
    import app.server as srv
    monkeypatch.setattr(srv, "_start_analysis", lambda *a, **kw: None)

    csv_bytes = b"price,size\n100,50\n200,80\n"
    resp = client.post(
        "/analyze",
        files={"file": ("housing.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"prompt": "predict house price", "level": "general", "tune": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "task_id" in body
    assert len(body["task_id"]) > 0
