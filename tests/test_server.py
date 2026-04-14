import json
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


def test_progress_sse_streams_events(client):
    import queue as q_module
    import app.server as srv

    task_id = "sse-test-001"
    test_q = q_module.Queue()
    test_q.put(json.dumps({"node": "profiling", "status": "done"}))
    test_q.put(json.dumps({"node": "intent_routing", "status": "done"}))
    test_q.put(json.dumps({"event": "complete", "task_type": "regression",
                            "best_model": "XGBoost", "elapsed": 3}))
    test_q.put(None)
    srv.TASK_QUEUES[task_id] = test_q

    with client.stream("GET", f"/progress/{task_id}") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.read().decode()

    assert '"profiling"' in body
    assert '"intent_routing"' in body
    assert '"complete"' in body


def test_progress_unknown_task(client):
    resp = client.get("/progress/does-not-exist")
    assert resp.status_code == 404


def test_report_download(client, tmp_path):
    import app.server as srv

    task_id = "report-test-001"
    report_file = tmp_path / "analysis_report.md"
    report_file.write_text("# Report\nHello world", encoding="utf-8")
    srv.TASK_RESULTS[task_id] = {"report_path": str(report_file)}

    resp = client.get(f"/report/{task_id}")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert "Report" in resp.text


def test_report_unknown_task(client):
    resp = client.get("/report/no-such-task")
    assert resp.status_code == 404


def test_report_missing_file(client):
    import app.server as srv
    task_id = "report-test-002"
    srv.TASK_RESULTS[task_id] = {"report_path": "/nonexistent/path.md"}
    resp = client.get(f"/report/{task_id}")
    assert resp.status_code == 404
