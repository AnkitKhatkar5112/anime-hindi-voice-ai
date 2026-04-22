# Phase 4 — Multi-Language Expansion: Tasks

- [x] 1. Haryanvi dialect support
  - [x] 1.1 Test existing `scripts/dialects/hindi_to_haryanvi.py` converter
    - Run: `python scripts/dialects/hindi_to_haryanvi.py`
    - Verify: `मैं यहाँ हूँ` → `म्हैं यड़े सूं`, `बहुत अच्छा है` → `घणा बढ़िया सै`
    - **Done when:** All test sentences print correct Haryanvi substitutions
    - _Requirements: 2_
  - [x] 1.2 Wire dialect post-processing into `scripts/inference/run_pipeline.py`
    - Read `configs/languages.yaml` for `dialect_post_process` and `dialect_script` fields
    - Add Stage 4b after translation when `dialect_post_process: true`
    - Update dialect scripts to accept `--input` and `--output` argparse args (process full JSON)
    - **Done when:** `--lang hry` pipeline runs Stage 4b and output JSON has Haryanvi-dialect text
    - _Requirements: 1, 3_
  - [x] 1.3 Expand Haryanvi word map to 60+ entries in `scripts/dialects/hindi_to_haryanvi.py`
    - Add: common verbs (dekh, sun, aa, ja, kha), pronouns, question words, adjectives, greetings
    - Research: Haryanvi phrasebooks, dialect dictionaries, native speaker validation
    - Add `test_pairs` list with 10 sentence pairs (Hindi → expected Haryanvi) and assert all pass
    - **Done when:** Word map has 60+ entries, all 10 test pairs produce expected output
    - _Requirements: 2_

---

- [x] 2. Punjabi language pipeline
  - [x] 2.1 Check Coqui TTS for Punjabi model availability
    - Run: `python -c "from TTS.api import TTS; [print(m) for m in TTS().list_models() if 'pa' in m]"`
    - **If model exists:** update `configs/languages.yaml` entry for `pa` with correct `tts_model` path
    - **If no model:** implement Google TTS API fallback (`pip install gTTS`) in `tts_hindi.py`
    - _Requirements: 4_
  - [x] 2.2 Run end-to-end Punjabi pipeline and validate output
    - Run: `python scripts/inference/run_pipeline.py --input sample.mp4 --lang pa`
    - Verify Gurmukhi script appears correctly in `data/processed/transcript_pa.json`
    - **Done when:** `outputs/final_pa_dub.wav` exists and is intelligible Punjabi audio
    - _Requirements: 4_

---

- [ ] 3. Tamil language pipeline
  - [ ] 3.1 Add Tamil support following Punjabi pattern
    - Note: Tamil is Dravidian — Google Translate quality from Japanese is lower than Hindi
    - Add `--review-mode` flag to `translate.py` that flags segments below confidence threshold
    - Save flagged segments to `transcript_ta_review.json` for human review
    - _Requirements: 5_
  - [ ] 3.2 Run end-to-end Tamil pipeline and validate output
    - Run: `python scripts/inference/run_pipeline.py --input sample.mp4 --lang ta`
    - Verify `--review-mode` produces a review file flagging uncertain translations
    - **Done when:** `outputs/final_ta_dub.wav` exists; review file produced
    - _Requirements: 5_

---

- [ ] 4. Language config abstraction validation
  - [ ] 4.1 Validate that adding a new language requires only `configs/languages.yaml` edits
    - Add Bengali (`bn`) entry to `languages.yaml` with `tts_model`, `translation_code`, `status: planned`
    - Run: `python scripts/inference/run_pipeline.py --input sample.mp4 --lang bn`
    - Document any code change that was required (there should be none)
    - If a code change was needed, fix the abstraction
    - **Done when:** Bengali pipeline runs (or fails gracefully) with zero Python code changes
    - _Requirements: 1_
  - [ ] 4.2 Write `docs/adding_a_language.md` step-by-step guide
    - Document the YAML config fields, model availability check, dialect script wiring
    - **Done when:** `docs/adding_a_language.md` exists with clear instructions
    - _Requirements: 1_
