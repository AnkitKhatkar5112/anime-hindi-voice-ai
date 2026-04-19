# Phase 7 — Testing & CI: Tasks

- [ ] 1. Test infrastructure and fixture setup
  - [ ] 1.1 Create `tests/conftest.py` with shared session-scoped fixtures
    - Define `SAMPLE_CLIP = "tests/fixtures/sample_30s.mp4"` and `PROCESSED_DIR`
    - Add `run_stage1()` session fixture: extracts audio once for entire test session
    - Create `tests/fixtures/` directory
    - _Requirements: 2_
  - [ ] 1.2 Add `tests/fixtures/sample_30s.mp4` test clip
    - 30-second anime clip, committed to repo, ≤5 MB
    - **Done when:** `pytest tests/conftest.py` runs without error
    - _Requirements: 2_

---

- [ ] 2. Pipeline stage unit tests
  - [ ] 2.1 Create `tests/test_extract_audio.py` — 5 assertions
    - `test_output_exists()` — WAV file was created
    - `test_sample_rate()` — sample rate is 22050
    - `test_mono()` — audio array is 1D (mono)
    - `test_non_empty()` — duration > 1 second
    - `test_duration_reasonable()` — between 20 and 40 seconds (30s fixture ± 10s)
    - _Requirements: 1, 3_
  - [ ] 2.2 Create `tests/test_asr.py` — 4 assertions
    - `test_has_segments()` — at least one segment exists
    - `test_segment_keys()` — each segment has `start`, `end`, `text`
    - `test_timestamps_ordered()` — start times are sorted
    - `test_has_japanese_text()` — at least one segment contains Japanese characters (U+3000–U+9FFF)
    - Uses module-scoped fixture running ASR with `--model medium`
    - _Requirements: 1, 3_
  - [ ] 2.3 Create `tests/test_translate.py` — 4 assertions
    - `test_no_segments_dropped()` — output has ≥95% of input segments
    - `test_translation_fields()` — each segment has `text_translated`
    - `test_cleaned_field()` — each segment has `text_cleaned` (from Phase 1 Task 8)
    - `test_no_empty_translations()` — <10% of translations are empty
    - _Requirements: 1, 3_
  - [ ] 2.4 Create `tests/test_tts.py` — 4 assertions
    - `test_segments_manifest_exists()` — `segments.json` exists
    - `test_all_audio_files_exist()` — every referenced audio file exists
    - `test_audio_files_non_empty()` — first 5 files have duration > 0.1s
    - `test_stretch_ratio_field()` — every segment has `stretch_ratio` (from Phase 1 Task 10)
    - _Requirements: 1, 3_

---

- [ ] 3. Output validation tests
  - [ ] 3.1 Create `tests/test_align.py` — 3 assertions
    - `test_output_exists()` — final mix WAV exists
    - `test_duration_within_tolerance()` — output duration within 5% of source
    - `test_no_clipping()` — <0.1% of samples have amplitude > 0.99
    - _Requirements: 1, 3_
  - [ ] 3.2 Create `tests/test_subtitles.py` — 3 assertions
    - `test_srt_exists()` — SRT file exists
    - `test_srt_has_content()` — file is non-empty
    - `test_srt_format()` — first 5 blocks have valid SRT format (index, timestamp pattern `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text)
    - _Requirements: 1, 3_

---

- [ ] 4. Quality benchmark
  - [ ] 4.1 Create `scripts/evaluation/benchmark.py` for PESQ/STOI tracking
    - Accept `--pairs` (JSON list of `{"original": "...", "dubbed": "..."}`) and `--output` log path
    - Call `evaluate_quality.evaluate()` on each pair
    - Compute and print `avg_pesq` and `avg_stoi`
    - Save timestamped JSON report to `logs/benchmark_YYYY-MM-DD.json`
    - Print summary table: pairs evaluated, avg PESQ (range 1–4.5), avg STOI (range 0–1)
    - **Done when:** Script runs, prints table, saves timestamped JSON to `logs/`
    - _Requirements: 5_
  - [ ] 4.2 Verify full test suite passes within time budget
    - Run: `pytest tests/ -v`
    - **Done when:** All tests pass in under 5 minutes on CPU
    - _Requirements: 4_
