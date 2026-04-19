"""
api/main.py — FastAPI REST interface for the Anime Dub AI pipeline.
Run with: uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uuid
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json

app = FastAPI(
    title="Anime Dub AI",
    description="Japanese → Hindi (+ multi-language) anime dubbing API",
    version="0.1.0"
)

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

jobs: dict = {}


def run_pipeline_job(job_id: str, input_path: str, lang: str):
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["stage"] = "extraction"

    output_path = str(JOBS_DIR / job_id / f"final_{lang}_dub.wav")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/inference/run_pipeline.py",
        "--input", input_path,
        "--lang", lang,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["output_file"] = f"outputs/final_{lang}_dub.wav"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.stderr[-500:]
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/dub")
async def submit_dub_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lang: str = "hi"
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(job_dir / file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "lang": lang,
        "input_file": file.filename
    }

    background_tasks.add_task(run_pipeline_job, job_id, input_path, lang)

    return {"job_id": job_id, "status": "queued", "lang": lang}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    output_file = job.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_file,
        media_type="audio/wav",
        filename=f"dubbed_{job['lang']}_{job_id}.wav"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
