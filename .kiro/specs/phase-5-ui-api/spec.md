# Spec: Phase 5 — UI / API Layer

## Overview
Expose the dubbing pipeline via a REST API and a browser-based demo UI. These are already implemented — this phase is about running, testing, and hardening them.

**Prerequisite:** Phase 1 complete.

---

## Requirements

1. FastAPI server accepts video uploads, processes them in the background, and serves the result for download
2. API is stateless — job state is stored in-process dict (upgrade to Redis in production later)
3. Streamlit UI allows file upload, language selection, progress display, and audio playback in under 5 clicks
4. Batch processor handles a folder of episodes, skips already-processed ones, and writes a summary report
5. API returns meaningful error messages (not 500) when pipeline stages fail

---

## Tasks

### Task 1: Run the FastAPI Server
Start the server:

```bash
uvicorn api.main:app --reload --port 8000
```

Test all three endpoints with curl:

```bash
# 1. Submit a dubbing job
curl -X POST "http://localhost:8000/dub?lang=hi" \
  -F "file=@sample.mp4"
# Expected: {"job_id": "abc12345", "status": "queued", "lang": "hi"}

# 2. Poll status
curl http://localhost:8000/status/abc12345
# Expected: {"job_id": "...", "status": "processing" | "complete" | "failed"}

# 3. Download result
curl -O http://localhost:8000/download/abc12345
# Expected: saves dubbed WAV file
```

Also test the health endpoint: `curl http://localhost:8000/health`

**Done when:** All three endpoints respond correctly. A submitted job completes and the result is downloadable.

---

### Task 2: Improve API Error Handling
Edit `api/main.py`. Currently, if the pipeline fails, the job silently sets `status: "failed"`. Improve this.

Changes:
1. The `/status/{job_id}` response should include `"error_summary"` when status is `"failed"` — a short human-readable description of which stage failed (not a raw traceback)
2. The `/download/{job_id}` endpoint should return HTTP 422 (not 400) with a clear JSON body when the job failed
3. Add a `GET /jobs` endpoint that lists all jobs with their statuses (useful for debugging)
4. Add request validation: reject uploads larger than 500 MB with HTTP 413

**Done when:** Failed jobs return actionable error messages. Large files are rejected with a clear response.

---

### Task 3: Run the Streamlit UI
Start the UI:

```bash
streamlit run ui/app.py
```

Opens at http://localhost:8501

Manual test flow:
1. Upload a 30-second `.mp4` clip
2. Select "Hindi (हिन्दी)" from the language dropdown
3. Click "Start Dubbing"
4. Wait for processing
5. Verify Hindi audio plays in the browser
6. Download the result

**Done when:** A non-technical user can complete the above flow in under 5 clicks with no errors.

---

### Task 4: Add Per-Stage Progress to Streamlit UI
Edit `ui/app.py`. The current UI shows a generic spinner. Improve it to show which pipeline stage is currently running.

Implementation approach:
- Write current stage name to a temp file (e.g. `logs/current_stage.txt`) from `run_pipeline.py`
- Streamlit reads and displays this file with `st.empty()` and `time.sleep(1)` polling

Or, alternatively, restructure the UI to call each stage's subprocess directly and update `st.progress()` bar after each one completes.

Add a collapsible "Show logs" expander that displays the last 20 lines of pipeline stdout.

**Done when:** UI shows the current stage name while processing (e.g. "🇯🇵 Transcribing Japanese..."). Progress bar increments stage by stage.

---

### Task 5: Run Batch Processing
Test the batch processor on a folder with 3+ episode files.

```bash
mkdir episodes/
# Copy 3 short test clips into episodes/

python scripts/inference/batch_process.py \
  --input-dir episodes/ \
  --lang hi \
  --skip-processed \
  --report logs/batch_report.json
```

Verify:
- `logs/batch_report.json` shows per-episode `"success": true/false`
- Running again with `--skip-processed` skips already-completed episodes
- A failed episode doesn't stop the rest from processing

**Done when:** Batch report shows correct per-episode status. `--skip-processed` works correctly.

---

## Acceptance Criteria

- [ ] FastAPI server starts with `uvicorn api.main:app --port 8000`
- [ ] `POST /dub` — accepts `.mp4` upload, returns `job_id`
- [ ] `GET /status/{job_id}` — returns current status and `error_summary` on failure
- [ ] `GET /download/{job_id}` — returns dubbed WAV file when status is `complete`
- [ ] `GET /jobs` — lists all jobs
- [ ] Files > 500 MB rejected with HTTP 413
- [ ] Streamlit UI — complete dub flow in under 5 clicks, audio plays in browser
- [ ] Streamlit UI — shows current stage name during processing
- [ ] `logs/batch_report.json` — correct per-episode success/failure after batch run
- [ ] `--skip-processed` — skips episodes with existing output files
