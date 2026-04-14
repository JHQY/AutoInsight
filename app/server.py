# app/server.py
import os
import sys
import uuid
import queue
import tempfile
import threading
import json
import time
import asyncio
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="AutoInsight")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# task_id -> queue.Queue (sentinel: None)
TASK_QUEUES: dict[str, queue.Queue] = {}
# task_id -> final AgentState dict
TASK_RESULTS: dict[str, dict] = {}
# task_id -> temp dir path
TASK_TMPDIRS: dict[str, str] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    target: str = Form(default=""),
    level: str = Form(default="general"),
    tune: str = Form(default="false"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    task_id = str(uuid.uuid4())

    tmp_dir = tempfile.mkdtemp(prefix=f"autoinsight_{task_id}_")
    TASK_TMPDIRS[task_id] = tmp_dir
    file_path = os.path.join(tmp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    TASK_QUEUES[task_id] = queue.Queue()

    tune_bool = tune.lower() in ("true", "1", "yes")
    initial_state = {
        "user_query":    prompt,
        "file_path":     file_path,
        "target_column": target,
        "user_level":    level,
        "tune":          tune_bool,
        "logs":          [],
        "schema":               {},
        "quality_issues":       [],
        "task_category":        "",
        "task_type":            "",
        "user_intent_summary":  "",
        "selected_algorithms":  [],
        "reasoning":            "",
        "model_results":        {},
        "metrics":              {},
        "best_model":           "",
        "charts":               [],
        "eda_summary":          {},
        "modeling_hints":       {},
        "feature_names":        [],
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "report_path":          "",
    }

    _start_analysis(task_id, initial_state)
    return {"task_id": task_id}


def _start_analysis(task_id: str, initial_state: dict) -> None:
    t = threading.Thread(target=_run_graph, args=(task_id, initial_state), daemon=True)
    t.start()


def _run_graph(task_id: str, initial_state: dict) -> None:
    q = TASK_QUEUES[task_id]
    t0 = time.time()
    try:
        from app.graph import build_graph
        graph = build_graph()
        final_state = dict(initial_state)
        for chunk in graph.stream(initial_state):
            for node_name, state_update in chunk.items():
                if isinstance(state_update, dict):
                    final_state.update(state_update)
                q.put(json.dumps({"node": node_name, "status": "done"}))
        elapsed = int(time.time() - t0)
        TASK_RESULTS[task_id] = final_state
        q.put(json.dumps({
            "event":      "complete",
            "task_type":  final_state.get("task_type", ""),
            "best_model": final_state.get("best_model", "") or "N/A",
            "elapsed":    elapsed,
        }))
    except Exception as exc:
        q.put(json.dumps({"event": "error", "message": str(exc)}))
    finally:
        q.put(None)
