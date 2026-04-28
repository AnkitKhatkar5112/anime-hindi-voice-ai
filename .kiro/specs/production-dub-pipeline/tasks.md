# Implementation Tasks: Production Dub Pipeline

## Task Status Legend
- `[ ]` Not started
- `[-]` In progress
- `[x]` Complete
- `[*]` Optional task

---

## Task 1: Core Data Models and Shared Utilities
*Req 2, 4, 5, 7, 12*

- [x] 1.1 Define `Segment` dataclass in `scripts/inference/models.py` with all fields: `segment_id`, `start`, `end`, `speaker_id`, `source_text`, `translated_text`, `dubbed_text`, `emotion`, `emotion_intensity`, `voice_profile`, `tts_audio_path`, `stretch_ratio`, `discarded`, `blended`, `overlap_gain`, `subtitle_type`
- [x] 1.2 Define `VoiceProfile` dataclass with fields: `profile_id`, `display_name`, `tts_backend`, `model_path`, `speaker_id`, `pitch_offset`, `rate_multiplier`
- [x] 1.3 Define `VoiceMap` schema (dataclass or TypedDict) including `tts_backend` key; add `load_voice_map()` and `save_voice_map()` helpers
- [x] 1.4 Define `PipelineResumeError` and `PipelineValidationError` custom exceptions in `scripts/inference/exceptions.py`
- [x] 1.5 Write unit tests for `Segment` construction, default values, and field types

---

## Task 2: Pipeline Cache Layer
*Req 16*

- [x] 2.1 Create `scripts/inference/pipeline_cache.py` with `PipelineCache` class
- [x] 2.2 Implement `cache_key(*parts: str) -> str` using `hashlib.sha256`
- [x] 2.3 Implement `get_translation(src_text, src_lang, tgt_lang, backend) -> dict | None` — checks `data/cache/translations/{key}.json`; logs `[CACHE HIT] translation {key[:8]}` on hit
- [x] 2.4 Implement `set_translation(...)` — writes JSON to `data/cache/translations/{key}.json`
- [x] 2.5 Implement `get_embedding(audio_path, mtime) -> np.ndarray | None` — checks `data/cache/embeddings/{key}.npy`
- [x] 2.6 Implement `set_embedding(...)` — writes `.npy` to `data/cache/embeddings/{key}.npy`
- [x] 2.7 Implement `get_tts(dubbed_text, voice_profile_id, emotion, emotion_intensity, tts_backend) -> str | None` — returns cached WAV path from `data/cache/tts/{key}.wav`
- [x] 2.8 Implement `set_tts(...)` — copies WAV to `data/cache/tts/{key}.wav`
- [x] 2.9 Implement `clear_cache() -> int` — deletes all files in all three subdirs, returns count deleted
- [x] 2.10 Write property-based test: for any combination of (text, lang, backend), `cache_key` output is deterministic and unique across distinct inputs (PBT — cache key uniqueness)

---

## Task 3: Failure Recovery — Checkpoint and Resume
*Req 17*

- [x] 3.1 Create `scripts/inference/checkpoint.py` with `CheckpointManager` class
- [x] 3.2 Implement `write_checkpoint(episode_id, completed_stages, input_file)` — writes `data/processed/{episode_id}_checkpoint.json` with schema `{episode_id, completed_stages, last_stage, timestamp, input_file}`
- [x] 3.3 Implement `read_checkpoint(episode_id) -> dict | None` — returns checkpoint dict or None if not found
- [x] 3.4 Implement `determine_start_stage(episode_id, resume_from=None, force_restart=False) -> int` — returns stage number per resume logic
- [x] 3.5 Implement `validate_stage_inputs(stage, episode_id)` — checks required input files exist; raises `PipelineResumeError` with list of missing files if any are absent
- [x] 3.6 Define `STAGE_INPUTS` dict mapping stage numbers 2–10 to their required input file path templates
- [x] 3.7 Write unit tests for `determine_start_stage`: (a) no checkpoint → stage 1, (b) checkpoint with last_stage=3 → stage 4, (c) `--resume-from 5` → stage 5, (d) `--force-restart` → stage 1
- [x] 3.8 Write unit test for `validate_stage_inputs`: missing file raises `PipelineResumeError` with descriptive message

---

## Task 4: Stage 1 — Ingest (SRT Parser + Audio Extraction)
*Req 2*

- [ ] 4.1 Create `scripts/preprocessing/srt_parser.py` with `SRTParser` class
- [ ] 4.2 Implement `parse(path: str, offset_ms: int = 0) -> list[Segment]` — reads UTF-8 and UTF-8-BOM `.srt` files using `pysrt` or custom parser
- [ ] 4.3 Apply `offset_ms` to all `start`/`end` times, clamping to minimum 0ms
- [ ] 4.4 Return error (empty list + log) if zero valid entries parsed; pipeline falls back to Whisper ASR
- [ ] 4.5 Implement `serialize(segments: list[Segment]) -> str` — produces valid SRT string with sequential index, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timecodes, UTF-8
- [ ] 4.6 Log applied offset and segment count adjusted
- [ ] 4.7 Write property-based test: for any valid list of Segments, `parse(serialize(segments))` produces timecodes within 10ms of originals (PBT — SRT round-trip, Req 2.6, 9.3)

---

## Task 5: Stage 2 — Audio Source Separation
*Req 6*

- [ ] 5.1 Create `scripts/preprocessing/separate_audio.py` with `SourceSeparator` class
- [ ] 5.2 Implement `separate(input_wav: str) -> tuple[str, str]` — runs Demucs `htdemucs` model, returns `(vocals_path, background_path)`
- [ ] 5.3 Implement SNR check: compute `snr_db = 10 * log10(signal_power / noise_power)` on vocals stem
- [ ] 5.4 Implement ducking fallback: if `snr_db < 10.0`, set `fallback_triggered=True`; use full extracted audio as background with volume 0.40 during dialogue, 1.00 outside
- [ ] 5.5 Write `data/processed/separation_quality.json` with `{snr_db, fallback_triggered, model}`
- [ ] 5.6 Log warning when fallback is triggered
- [ ] 5.7 Write unit test: if Demucs not installed, `SourceSeparator` logs warning and returns `(extracted.wav, extracted.wav)` without raising

---

## Task 6: Stage 3 — Speaker Diarization with Fast-Mode
*Req 4.7–4.9*

- [ ] 6.1 Extend `scripts/preprocessing/diarize_speakers.py` to accept `--fast-mode` flag
- [ ] 6.2 WHEN `--fast-mode` is set: skip pyannote entirely; call `Voice_Embedding_Clusterer` (Resemblyzer) as the only method
- [ ] 6.3 WHEN `--fast-mode` is NOT set: use pyannote as primary (existing behavior)
- [ ] 6.4 Implement `Voice_Embedding_Clusterer.cluster(segments, vocals_wav) -> list[Segment]` — extracts Resemblyzer embeddings per segment, runs agglomerative clustering (cosine distance threshold 0.25), assigns `SPEAKER_XX` IDs
- [ ] 6.5 Integrate embedding cache: call `PipelineCache.get_embedding(audio_path, mtime)` before computing; call `set_embedding` after
- [ ] 6.6 Log active diarization mode at startup: `"Diarization mode: fast (Resemblyzer)"` or `"Diarization mode: standard (pyannote)"`
- [ ] 6.7 Write unit test: `--fast-mode` with a 3-speaker audio fixture produces 3 distinct `SPEAKER_XX` IDs

---

## Task 7: Stage 4 — Translation and Dubbing Adapter
*Req 3*

- [ ] 7.1 Update `scripts/preprocessing/translate.py` to use `Helsinki-NLP/opus-mt-ja-hi` as primary model
- [ ] 7.2 Implement fallback chain: DeepL API → Google Translate, used only if Helsinki-NLP raises an exception
- [ ] 7.3 Lock translation backend at episode start: detect once, log `"Translation backend: {name}"`, pass locked backend to all segment translations
- [ ] 7.4 Integrate translation cache: call `PipelineCache.get_translation(...)` before translating; call `set_translation(...)` after
- [ ] 7.5 Create `scripts/preprocessing/dubbing_adapter.py` with `DubbingAdapter` class
- [ ] 7.6 Implement backend selection: (1) GPT-4o if `OPENAI_API_KEY` set, (2) Ollama if reachable, (3) rule-based fallback
- [ ] 7.7 Implement rule-based fallback: honorific substitution table (senpai→सीनियर, nakama→साथी, etc.) + length trimming to `duration_s × 14 × 1.3` chars
- [ ] 7.8 Implement GPT-4o adapter: system prompt enforcing conversational Hindi, honorific localization, no Japanese characters, character count limit
- [ ] 7.9 Implement Ollama adapter: POST to local Ollama API with same system prompt
- [ ] 7.10 Log selected Dubbing_Adapter backend at startup
- [ ] 7.11 Validate output: raise if translated text contains Hiragana/Katakana/Kanji (`\u3040-\u30ff\u4e00-\u9fff`)
- [ ] 7.12 Write property-based test: for any segment with `duration_s` and dubbed text, `len(dubbed_text) <= duration_s * 14 * 1.3` (PBT — character count constraint, Req 3.3)

---

## Task 8: Stage 5 — Emotion Detection
*Req 5.11–5.13*

- [ ] 8.1 Extend `scripts/preprocessing/detect_emotion.py` with two-stage strategy
- [ ] 8.2 Implement rule-based stage: scan for `!`, `?`, Hindi keywords (`nahi`, `kyu`, `bachao`, `ruko`, `haha`, `wah`); map to `{neutral, happy, angry, sad, excited, fearful}` + base intensity
- [ ] 8.3 Implement AI enrichment stage: run `transformers` emotion classifier; override rule-based result only if confidence > 0.7
- [ ] 8.4 Implement fallback: if AI model fails or is unavailable, keep rule-based result; always produce valid `emotion` label
- [ ] 8.5 Ensure every segment gets `emotion` from the valid set and `emotion_intensity` in `[0.0, 1.0]`
- [ ] 8.6 Write unit test: segment with `"nahi!"` text gets `emotion="angry"` from rule-based stage

---

## Task 9: Stage 6 — Voice Assignment and TTS Backend Locking
*Req 4, 5.7–5.8*

- [ ] 9.1 Create `scripts/inference/character_voice_mapper.py` with `CharacterVoiceMapper` class
- [ ] 9.2 Implement `assign(segments, episode_id, shared_voice_map=None) -> list[Segment]` — loads existing map if present, assigns new speakers round-robin from `character_voices.yaml`
- [ ] 9.3 Implement TTS backend locking: on first run, detect backend (NeMo → Coqui → gTTS), write `tts_backend` key to voice map JSON
- [ ] 9.4 On resume: if voice map exists with `tts_backend` key, reuse that backend without re-detecting
- [ ] 9.5 Log TTS backend selected/reused at startup
- [ ] 9.6 Implement `--shared-voice-map` support: load cross-episode map and extend it
- [ ] 9.7 Log warning when speaker count exceeds available profiles (cycling behavior)
- [ ] 9.8 Write unit test: same `episode_id` across two runs produces identical speaker→voice assignments (persistence test)

---

## Task 10: Stage 7 — TTS Synthesis with Emotion Prosody
*Req 5*

- [ ] 10.1 Extend `scripts/inference/tts_hindi.py` to accept `VoiceProfile`, `emotion`, `emotion_intensity` per segment
- [ ] 10.2 Implement prosody application: rate and pitch modifiers per emotion table (angry/sad/happy/excited/neutral)
- [ ] 10.3 Implement Bark routing: if `emotion_intensity > 0.7`, route to Bark TTS with emotion-matched Hindi preset
- [ ] 10.4 Implement time-stretch: if TTS output exceeds segment duration, apply `librosa.effects.time_stretch` (ratio ≤ 1.4×)
- [ ] 10.5 Implement retry: if stretch insufficient, shorten dubbed text and re-synthesize (max 1 retry)
- [ ] 10.6 Integrate TTS cache: call `PipelineCache.get_tts(...)` before synthesizing; call `set_tts(...)` after; log `[CACHE HIT] tts {key[:8]}` on hit
- [ ] 10.7 Write unit test: `emotion_intensity=0.8` routes to Bark backend
- [ ] 10.8 Write unit test: TTS output longer than segment duration triggers time-stretch

---

## Task 11: Stage 8 — Overlap Resolution and Audio Mixing
*Req 7, 12*

- [ ] 11.1 Extend `scripts/inference/align_and_mix.py` with overlap detection loop
- [ ] 11.2 Implement trim strategy: overlap < 200ms → set `seg_i.end = seg_j.start`
- [ ] 11.3 Implement blend strategy: overlap ≥ 200ms → set `lower_priority.blended = True`, `lower_priority.overlap_gain = 0.40`; both segments preserved
- [ ] 11.4 Apply `overlap_gain` during mixing: when placing a blended segment on the timeline, multiply its audio by `overlap_gain` during the overlap window only
- [ ] 11.5 Implement LUFS mixing pipeline: 10ms fade-in/fade-out per segment, blend background (0.20) + dialogue (0.85), measure LUFS with `pyloudnorm`, normalize to -16 LUFS, apply -1.0 dBFS peak ceiling
- [ ] 11.6 Log pre- and post-normalization LUFS values
- [ ] 11.7 Write `logs/mixing_report_{episode_id}.json` including `blended` field per segment
- [ ] 11.8 Write property-based test: for any set of non-overlapping segments, post-normalization LUFS is within 0.5 dB of -16 LUFS (PBT — LUFS normalization target, Req 7.1)
- [ ] 11.9 Write property-based test: for any two overlapping segments (overlap ≥ 200ms), both segments appear in the mixing report with `discarded: false` (PBT — overlap blend correctness, Req 12.4)

---

## Task 12: Stage 8b — Subtitle Processing Module
*Req 15*

- [ ] 12.1 Create `scripts/inference/subtitle_processor.py` with `SubtitleClassifier`, `SubtitleSplitter`, `SignTranslator`, `SubtitleWriter` classes
- [ ] 12.2 Implement `SubtitleClassifier.classify(text) -> str` — returns `"sign"`, `"dialogue"`, or `"mixed"` using SIGN_PATTERNS and SOUND_EFFECTS rules
- [ ] 12.3 Implement `SubtitleSplitter.split(segment) -> tuple[Segment, Segment]` — splits mixed segments into sign + dialogue using regex
- [ ] 12.4 Implement `SignTranslator.translate(text) -> str` — calls translation backend with `literal=True`, bypasses DubbingAdapter
- [ ] 12.5 Implement `SubtitleWriter.write_srt(segments, output_path)` — sequential index, `HH:MM:SS,mmm` timecodes, UTF-8, sign text wrapped in `[...]`
- [ ] 12.6 [*] Implement `SubtitleWriter.write_ass(segments, output_path)` — sign subtitles at top (`\an8`), dialogue at bottom; Noto Sans Devanagari font style block
- [ ] 12.7 Write unit test: ALL CAPS text classifies as `"sign"`; mixed-case sentence classifies as `"dialogue"`
- [ ] 12.8 Write unit test: `"[BANG] Character speaks here"` classifies as `"mixed"` and splits correctly

---

## Task 13: Stage 8b — OCR Engine (Optional)
*Req 15.15–15.20*

- [ ] 13.1 Create `scripts/inference/ocr_engine.py` with `OCREngine` class
- [ ] 13.2 Implement frame extraction: `ffmpeg -i {video} -vf fps=1 data/processed/frames/frame_%04d.jpg`
- [ ] 13.3 Implement OCR: run `easyocr.Reader(['ja'])` on each frame; filter to top 80% of frame height
- [ ] 13.4 Implement `align_ocr_timestamp(raw_ts, segment_boundaries) -> float` — finds nearest SRT/ASR boundary within ±500ms; uses raw timestamp if none found
- [ ] 13.5 Log aligned vs. raw count: `"[OCR] {aligned} signs aligned to SRT boundary, {raw} used raw timestamp"`
- [ ] 13.6 Translate detected sign text (literal) and append to subtitle segment list with `start=aligned_ts`, `end=aligned_ts + 2.0`
- [ ] 13.7 Write `data/processed/ocr_signs.json` with detected regions
- [ ] 13.8 Write unit test for `align_ocr_timestamp`: timestamp 1.3s with boundary at 1.1s → returns 1.1s; no boundary within ±500ms → returns raw timestamp

---

## Task 14: Stage 9 — FFmpeg Muxer
*Req 1*

- [ ] 14.1 Create `scripts/inference/mux_output.py` with `Muxer` class
- [ ] 14.2 Implement `mux(input_video, hindi_wav, output_path)` — builds and runs FFmpeg subprocess command
- [ ] 14.3 MP4 command: `-map 0:v -map 0:a -map 1:a -c:v copy -c:a:0 copy -c:a:1 aac -b:a:1 192k -metadata:s:a:0 language=jpn -metadata:s:a:0 title="Japanese" -metadata:s:a:1 language=hin -metadata:s:a:1 title="Hindi" -disposition:a:0 default`
- [ ] 14.4 MKV command: same but `-c:a:1 libopus -b:a:1 128k`
- [ ] 14.5 Log exact FFmpeg command used to pipeline log
- [ ] 14.6 Post-mux: verify output file exists and `os.path.getsize > 0`; raise on failure
- [ ] 14.7 Write unit test: mux with a fixture video + WAV produces output file with correct extension matching input

---

## Task 15: Stage 10 — Output Validation
*Req 13*

- [ ] 15.1 Create `scripts/inference/validate_output.py` with `Validator` class
- [ ] 15.2 Implement `validate(output_path, episode_id) -> dict` — runs FFprobe JSON, checks: audio track count == 2, duration within 2s, Hindi language tag == `"hin"`, file size > 0
- [ ] 15.3 Write `logs/output_validation_{episode_id}.json` with full FFprobe output + per-check pass/fail
- [ ] 15.4 Raise `PipelineValidationError` listing failed checks if any check fails
- [ ] 15.5 Write unit test: FFprobe output with 1 audio stream raises `PipelineValidationError` with `"audio_track_count"` in message

---

## Task 16: Pipeline Orchestrator — run_pipeline.py
*Req 4.7–4.9, 16, 17*

- [ ] 16.1 Add CLI arguments to `scripts/inference/run_pipeline.py`:
  - `--fast-mode` (flag)
  - `--resume-from N` (int, 1–10)
  - `--force-restart` (flag)
  - `--clear-cache` (flag)
  - `--subtitle-offset N` (int, ms)
  - `--subtitle-format {srt,ass}` (default: srt)
  - `--ocr-signs` (flag)
  - `--shared-voice-map PATH`
- [ ] 16.2 Implement `--clear-cache` early exit: call `PipelineCache.clear_cache()`, log count, exit 0
- [ ] 16.3 Integrate `CheckpointManager.determine_start_stage()` at startup to determine which stage to begin from
- [ ] 16.4 Wrap each stage in try/except: on success, call `CheckpointManager.write_checkpoint()`; on failure, log exception and exit
- [ ] 16.5 Call `validate_stage_inputs(stage, episode_id)` before each stage; catch `PipelineResumeError` and print descriptive message
- [ ] 16.6 Log diarization mode, translation backend, and TTS backend at startup
- [ ] 16.7 Pass `--fast-mode` flag to diarization stage
- [ ] 16.8 Write integration test: pipeline with `--resume-from 4` skips stages 1–3 and loads intermediate outputs from fixture files

---

## Task 17: Pipeline Progress Logging
*Req 14*

- [ ] 17.1 Create `scripts/inference/log_writer.py` with `LogWriter` class
- [ ] 17.2 Implement `start_stage(stage_name, input_paths)` — writes timestamped entry to `logs/pipeline_{episode_id}_{timestamp}.log` and overwrites `logs/current_stage.txt`
- [ ] 17.3 Implement `end_stage(stage_name, output_paths, segment_count, duration_s)` — writes completion entry with duration
- [ ] 17.4 Implement `log_exception(stage_name, exc)` — writes stage name, exception type, message, and full traceback
- [ ] 17.5 Write unit test: `current_stage.txt` contains current stage name while stage is executing

---

## Task 18: Training Data Pipeline
*Req 8*

- [ ] 18.1 Verify `training/` folder structure exists: `source/`, `dubbed/`, `wavs/`, `metadata.csv`
- [ ] 18.2 Extend `scripts/training/prepare_dataset.py`: validate each `metadata.csv` entry has a corresponding file in `wavs/`; report missing files
- [ ] 18.3 Create `scripts/training/dataset_stats.py`: report total audio duration, speaker count, avg segment length, vocabulary coverage
- [ ] 18.4 [*] Create `scripts/training/finetune_nemo.py`: fine-tune NeMo FastPitch/VITS from LJ-Speech dataset using `--base-model` argument

---

## Task 19: API and UI Updates
*Req 10*

- [ ] 19.1 Update `api/main.py` `POST /dub` endpoint: add optional `srt_file: UploadFile = None` and `subtitle_offset: int = 0` params
- [ ] 19.2 Update `GET /download/{job_id}`: return muxed video with correct `Content-Type` (`video/mp4` or `video/x-matroska`)
- [ ] 19.3 Add file format validation: accept `.mp4`, `.mkv`, `.avi`; return HTTP 422 for others
- [ ] 19.4 Add `GET /voices` endpoint: return `configs/character_voices.yaml` as JSON
- [ ] 19.5 Update `ui/app.py`: add second `st.file_uploader` for optional `.srt` file
- [ ] 19.6 Add `st.number_input` for `--subtitle-offset` (ms, default 0, range -300 to +300)
- [ ] 19.7 Add download button for muxed video on job completion
- [ ] 19.8 Poll `logs/current_stage.txt` for real-time stage display in UI

---

## Task 20: Property-Based Tests Summary
*Cross-cutting*

- [ ] 20.1 **SRT round-trip** (Req 2.6, 9.3): `parse(serialize(segments))` timecodes within 10ms — implemented in Task 4.7
- [ ] 20.2 **Cache key uniqueness** (Req 16): distinct inputs produce distinct SHA256 keys — implemented in Task 2.10
- [ ] 20.3 **LUFS normalization target** (Req 7.1): post-normalization LUFS within 0.5 dB of -16 LUFS — implemented in Task 11.8
- [ ] 20.4 **Overlap blend correctness** (Req 12.4): no segment discarded for overlap ≥ 200ms — implemented in Task 11.9
- [ ] 20.5 **Character count constraint** (Req 3.3): dubbed text length ≤ `duration_s × 14 × 1.3` — implemented in Task 7.12
- [ ] 20.6 Configure all property tests to run minimum 100 iterations (use `hypothesis` settings or `pytest-randomly` seed)
- [ ] 20.7 Tag each property test with comment: `# Feature: production-dub-pipeline, Property N: {property_text}`

---

## Dependency Order

```
Task 1 (models)
  └── Task 2 (cache)
  └── Task 3 (checkpoint)
  └── Task 4 (SRT parser)
        └── Task 5 (source separation)
              └── Task 6 (diarization)
                    └── Task 7 (translation)
                          └── Task 8 (emotion)
                                └── Task 9 (voice assignment)
                                      └── Task 10 (TTS)
                                            └── Task 11 (mixing)
                                                  └── Task 12 (subtitles)
                                                  └── Task 13* (OCR)
                                                        └── Task 14 (muxer)
                                                              └── Task 15 (validation)
Task 16 (orchestrator) — depends on Tasks 2, 3, 6, 7, 9, 10, 11, 12, 14, 15
Task 17 (logging) — can be built in parallel with Tasks 4–15
Task 18 (training) — independent
Task 19 (API/UI) — depends on Task 16
Task 20 (PBT summary) — depends on Tasks 4, 2, 11, 7
```
