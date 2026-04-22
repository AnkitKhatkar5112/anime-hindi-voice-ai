"""
api/main.py — FastAPI REST interface for the Anime Dub AI pipeline.
Run with: uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
import uuid
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import json

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

app = FastAPI(
    title="Anime Dub AI",
    description="Japanese → Hindi (+ multi-language) anime dubbing API",
    version="0.1.0"
)

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

jobs: dict = {}


def _extract_error_summary(stderr: str) -> str:
    """Return a short human-readable summary of which pipeline stage failed."""
    stage_keywords = {
        "extract": "Audio extraction failed",
        "diariz": "Speaker diarization failed",
        "transcri": "Transcription (ASR) failed",
        "translat": "Translation failed",
        "tts": "Text-to-speech synthesis failed",
        "align": "Audio alignment failed",
        "lip": "Lip-sync failed",
    }
    lower = stderr.lower()
    for keyword, label in stage_keywords.items():
        if keyword in lower:
            return label
    # Fall back to last non-empty line of stderr
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    return lines[-1][:200] if lines else "Pipeline failed (unknown stage)"


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
            jobs[job_id]["error_summary"] = _extract_error_summary(result.stderr)
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["error_summary"] = str(e)


@app.post("/dub")
async def submit_dub_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    lang: str = "hi"
):
    # Reject uploads larger than 500 MB
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is 500 MB."
        )

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Stream to disk while enforcing size limit
    input_path = str(job_dir / file.filename)
    bytes_written = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                f.close()
                Path(input_path).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="File too large. Maximum allowed size is 500 MB."
                )
            f.write(chunk)

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
    job = jobs[job_id]
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "lang": job.get("lang"),
        "input_file": job.get("input_file"),
    }
    if job["status"] == "failed":
        response["error_summary"] = job.get("error_summary", "Pipeline failed")
    if job["status"] == "processing":
        response["stage"] = job.get("stage")
    return response


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Job failed — cannot download result.",
                "error_summary": job.get("error_summary", "Pipeline failed"),
                "job_id": job_id,
            }
        )
    if job["status"] != "complete":
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Job is not complete yet.",
                "status": job["status"],
                "job_id": job_id,
            }
        )

    output_file = job.get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_file,
        media_type="audio/wav",
        filename=f"dubbed_{job['lang']}_{job_id}.wav"
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs with their current statuses."""
    return [
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "lang": job.get("lang"),
            "input_file": job.get("input_file"),
            **({"error_summary": job["error_summary"]} if job["status"] == "failed" else {}),
        }
        for job in jobs.values()
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
