# Spec: Phase 4 — Multi-Language Expansion

## Overview
Extend the pipeline to support Indian regional languages beyond Hindi. The architecture uses `configs/languages.yaml` as the single source of truth — adding a language should require no code changes, only config updates and model availability.

**Prerequisite:** Phase 1 complete. Hindi pipeline works end-to-end.

---

## Requirements

1. `configs/languages.yaml` is the sole place to add a new language — no code changes needed
2. Haryanvi works via Hindi TTS + rule-based dialect post-processor (`scripts/dialects/hindi_to_haryanvi.py`)
3. `run_pipeline.py` automatically reads `languages.yaml` and applies dialect post-processing when `dialect_post_process: true`
4. Punjabi pipeline runs end-to-end with `--lang pa`
5. Tamil pipeline runs end-to-end with `--lang ta`

---

## Tasks

### Task 1: Test Haryanvi Dialect Converter
Run the already-implemented converter to verify it works.

```bash
python scripts/dialects/hindi_to_haryanvi.py
```

Expected output — Hindi sentences converted to Haryanvi dialect:
- `मैं यहाँ हूँ` → `म्हैं यड़े सूं`
- `बहुत अच्छा है` → `घणा बढ़िया सै`

**Done when:** All test sentences in the script print correct Haryanvi substitutions.

---

### Task 2: Wire Dialect Post-Processing into run_pipeline.py
Edit `scripts/inference/run_pipeline.py` to read `configs/languages.yaml` and apply dialect scripts automatically.

After Stage 4 (translation), add:

```python
import yaml

with open("configs/languages.yaml") as f:
    lang_cfgs = yaml.safe_load(f)["languages"]

lang_cfg = lang_cfgs.get(args.lang, {})

if lang_cfg.get("dialect_post_process") and lang_cfg.get("dialect_script"):
    run_stage(
        f"Stage 4b: Dialect Post-Processing ({args.lang})",
        lang_cfg["dialect_script"],
        ["--input",  f"data/processed/transcript_{args.lang}.json",
         "--output", f"data/processed/transcript_{args.lang}.json"]
    )
```

Also update the dialect scripts (`hindi_to_haryanvi.py` etc.) to accept `--input` and `--output` argparse arguments and process the full JSON — not just print test sentences.

**Done when:** `python scripts/inference/run_pipeline.py --input sample.mp4 --lang hry` runs Stage 4b and the output JSON has Haryanvi-dialect text before TTS.

---

### Task 3: Expand Haryanvi Word Map
Edit `scripts/dialects/hindi_to_haryanvi.py` — the current substitution map has ~15 words. Expand it to at least 60 common substitutions.

Research sources:
- Haryanvi phrasebooks and dialect dictionaries
- Native speaker review (ask a Haryanvi speaker to validate additions)

Add at minimum: common verbs (dekh, sun, aa, ja, kha), pronouns, question words, common adjectives, and greetings.

Also add `test_pairs` list at the bottom with 10 sentence pairs (Hindi → expected Haryanvi) and assert all pass.

**Done when:** Word map has 60+ entries. All 10 test pairs produce expected Haryanvi output.

---

### Task 4: Add Punjabi Support
Check Coqui TTS for a Punjabi model:

```bash
python -c "from TTS.api import TTS; [print(m) for m in TTS().list_models() if 'pa' in m]"
```

**If a Coqui Punjabi model exists:**
- Update `configs/languages.yaml` entry for `pa` with the correct `tts_model` path
- Test: `python scripts/inference/run_pipeline.py --input sample.mp4 --lang pa`

**If no Coqui model:**
- Implement Google TTS API fallback in `scripts/inference/tts_hindi.py`
- Add a branch: if `lang_cfg.get("tts_engine") == "google"`, use `gtts` library instead of Coqui
- Install: `pip install gTTS`

Verify Gurmukhi script appears correctly in `data/processed/transcript_pa.json`.

**Done when:** `outputs/final_pa_dub.wav` exists and is intelligible Punjabi audio.

---

### Task 5: Add Tamil Support
Follow the same pattern as Task 4 for Tamil (`ta`).

Notes:
- Tamil is a Dravidian language — Google Translate quality from Japanese is lower than for Hindi
- Add a `--review-mode` flag to `translate.py` that saves a separate `transcript_ta_review.json` flagging segments where translation confidence is below a threshold
- This allows a human reviewer to fix bad translations before TTS

**Done when:** `outputs/final_ta_dub.wav` exists. `--review-mode` flag produces a review file flagging uncertain translations.

---

### Task 6: Validate Language Config Abstraction
Verify that adding a completely new language requires only editing `configs/languages.yaml`.

Test by adding Bengali (`bn`) to `languages.yaml`:
```yaml
bn:
  name: Bengali
  iso_code: bn
  tts_model: tts_models/bn/cv/vits
  translation_code: bn
  dialect_post_process: false
  dialect_script: null
  status: planned
```

Then run `python scripts/inference/run_pipeline.py --input sample.mp4 --lang bn`.

Document any code change that was required (there should be none). If a code change was required, fix the abstraction so it isn't.

**Done when:** Bengali pipeline runs (or fails gracefully with "model not found") with zero code changes. Any required changes are documented in `docs/adding_a_language.md`.

---

## Acceptance Criteria

- [ ] `python scripts/dialects/hindi_to_haryanvi.py` — all test sentence pairs pass
- [ ] `--lang hry` pipeline — Stage 4b runs and output has Haryanvi dialect text
- [ ] `hindi_to_haryanvi.py` — 60+ substitution entries
- [ ] `outputs/final_pa_dub.wav` — intelligible Punjabi audio
- [ ] `outputs/final_ta_dub.wav` — intelligible Tamil audio
- [ ] `--review-mode` flag on `translate.py` — produces a review JSON for Tamil
- [ ] Adding Bengali to `languages.yaml` requires zero code changes to Python files
- [ ] `docs/adding_a_language.md` — step-by-step guide written
