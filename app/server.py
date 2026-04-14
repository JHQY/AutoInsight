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
