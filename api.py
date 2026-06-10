"""Thin REST surface over the VERDICT index - the hosted-sandbox demo app.

  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints (all shell out to the audited CLIs - one code path, no drift):
  GET  /healthz                 index size + artifact freshness
  GET  /search?query=...        recruiter search (same engine as search.py)
  GET  /explain/{candidate_id}  full J x C x A breakdown with evidence
  POST /ingest                  upload JSON/JSONL candidates -> searchable immediately
  POST /rank?top=100            run the challenge ranking, returns CSV
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

ROOT = Path(__file__).parent
PY = sys.executable
ART = ROOT / "artifacts"

app = FastAPI(title="VERDICT", version="0.1.0")


def _run(args: list[str], timeout: int = 600) -> str:
    proc = subprocess.run(
        [PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=(proc.stderr or proc.stdout)[-2000:])
    return proc.stdout


@app.get("/healthz")
def healthz():
    ids_path = ART / "candidate_ids.txt"
    n = sum(1 for _ in ids_path.open(encoding="utf-8")) if ids_path.exists() else 0
    return {"status": "ok", "candidates_indexed": n,
            "index_updated": time.ctime(ids_path.stat().st_mtime) if n else None}


@app.get("/search")
def search(
    query: str = Query(..., description="plain-language search"),
    preset: str = Query("ai_ml"),
    top: int = Query(25, le=200),
    min_yoe: float = 0.0,
    max_yoe: float = 50.0,
    availability: bool = False,
    good_companies: bool = False,
    location: str = Query("any"),
):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = tmp.name
    args = ["search.py", "--query", query, "--preset", preset, "--top", str(top),
            "--min-yoe", str(min_yoe), "--max-yoe", str(max_yoe),
            "--location", location, "--out", out]
    if availability:
        args.append("--availability")
    if good_companies:
        args.append("--good-companies")
    _run(args)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    Path(out).unlink(missing_ok=True)
    return {"count": len(rows), "results": rows}


@app.get("/explain/{candidate_id}")
def explain(candidate_id: str):
    out = _run(["explain.py", candidate_id, "--json"])
    return JSONResponse(content=json.loads(out))


@app.post("/ingest")
async def ingest(file: UploadFile, replace: bool = False):
    data = await file.read()
    suffix = ".jsonl.gz" if file.filename.endswith(".gz") else ".jsonl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    args = ["ingest.py", "--input", path]
    if replace:
        args.append("--replace")
    log = _run(args)
    Path(path).unlink(missing_ok=True)
    return {"status": "ingested", "log": log.strip().splitlines()[-3:]}


@app.post("/rank")
def rank(top: int = Query(100, le=100), candidates: str | None = None):
    src = candidates or str(
        ROOT / "[PUB] India_runs_data_and_ai_challenge"
        / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = tmp.name
    _run(["rank.py", "--candidates", src, "--out", out, "--top", str(top)])
    text = Path(out).read_text(encoding="utf-8")
    Path(out).unlink(missing_ok=True)
    return PlainTextResponse(text, media_type="text/csv")
