# Phase 5 — UI / API Layer: Tasks

- [ ] 1. FastAPI server setup and endpoint testing
  - [ ] 1.1 Start the FastAPI server and test all endpoints
    - Start: `uvicorn api.main:app --reload --port 8000`
    - Test `POST /dub?lang=hi` with `curl -X POST -F "file=@sample.mp4"` → expect `{"job_id": "...", "status": "queued"}`
    - Test `GET /status/{job_id}` → expect `{"status": "processing" | "complete" | "failed"}`
    - Test `GET /download/{job_id}` → expect WAV file download
    - Test `GET /health` → expect health check response
    - **Done when:** All endpoints respond correctly; submitted job completes and result is downloadable
    - _Requirements: 1, 2_
  - [ ] 1.2 Improve API error handling in `api/main.py`
    - Add `"error_summary"` field to `/status/{job_id}` response when status is `"failed"`
    - Return HTTP 422 (not 400) with clear JSON body on `/download/{job_id}` for failed jobs
    - Add `GET /jobs` endpoint listing all jobs with their statuses
    - Add request validation: reject uploads > 500 MB with HTTP 413
    - **Done when:** Failed jobs return actionable error messages; large files rejected clearly
    - _Requirements: 5_

---

- [ ] 2. Streamlit UI testing and enhancement
  - [ ] 2.1 Run and manually test the Streamlit UI
    - Start: `streamlit run ui/app.py` → opens at http://localhost:8501
    - Test flow: Upload `.mp4` → Select "Hindi (हिन्दी)" → Click "Start Dubbing" → Wait → Play audio → Download
    - **Done when:** Non-technical user can complete the flow in under 5 clicks with no errors
    - _Requirements: 3_
  - [ ] 2.2 Add per-stage progress display to `ui/app.py`
    - Option A: Write current stage name to `logs/current_stage.txt` from `run_pipeline.py`, poll in Streamlit
    - Option B: Call each stage subprocess directly, update `st.progress()` after each completes
    - Add collapsible "Show logs" expander with last 20 lines of pipeline stdout
    - **Done when:** UI shows current stage name (e.g. "🇯🇵 Transcribing Japanese..."); progress bar increments per stage
    - _Requirements: 3_

---

- [ ] 3. Batch processing
  - [ ] 3.1 Test batch processor on a folder with 3+ episode files
    - Create `episodes/` folder with 3 short test clips
    - Run: `python scripts/inference/batch_process.py --input-dir episodes/ --lang hi --skip-processed --report logs/batch_report.json`
    - Verify `logs/batch_report.json` shows per-episode `"success": true/false`
    - Verify re-running with `--skip-processed` skips completed episodes
    - Verify a failed episode doesn't stop the rest from processing
    - **Done when:** Batch report shows correct status per episode; `--skip-processed` works
    - _Requirements: 4_
