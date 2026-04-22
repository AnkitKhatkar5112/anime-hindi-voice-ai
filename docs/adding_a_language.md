# Adding a New Language to the Anime Dub AI Pipeline

`configs/languages.yaml` is the single source of truth for language support.
No Python code changes are needed to add a new language.

---

## Step 1: Add an entry to `configs/languages.yaml`

```yaml
languages:
  xx:                          # ISO 639-1 code (e.g. bn, ta, te, mr)
    name: Language Name        # English name
    name_native: নাম           # Name in native script
    script: Script Name        # e.g. Bengali, Devanagari, Tamil
    iso_code: xx               # Must match the top-level key
    tts_model: tts_models/xx/cv/vits   # Coqui model path (or null if using Google TTS)
    tts_engine: null           # Set to "google" to force gTTS instead of Coqui
    translation_code: xx       # Google Translate target code (usually same as iso_code)
    dialect_post_process: false
    dialect_script: null       # Path to dialect script, e.g. scripts/dialects/xx_dialect.py
    status: planned            # planned | in_progress | active
```

### Field reference

| Field | Required | Description |
|---|---|---|
| `iso_code` | yes | ISO 639-1/2 code. Must match the YAML key. |
| `tts_model` | yes* | Coqui TTS model path. Set to `null` if using `tts_engine: google`. |
| `tts_engine` | no | Set to `"google"` to use gTTS instead of Coqui (e.g. for languages without a Coqui model). |
| `translation_code` | yes | Language code passed to Google Translate. Usually the same as `iso_code`. |
| `dialect_post_process` | yes | `true` to run a dialect post-processing script after translation. |
| `dialect_script` | no | Path to the dialect script (required when `dialect_post_process: true`). |
| `status` | yes | `planned` / `in_progress` / `active`. Informational only — does not affect pipeline execution. |

---

## Step 2: Check model availability

Before running the pipeline, verify whether a Coqui model exists for your language:

```bash
python -c "from TTS.api import TTS; [print(m) for m in TTS().list_models() if 'xx' in m]"
```

Replace `xx` with your language code. If no model is listed:
- Set `tts_model: null` and `tts_engine: google` in `languages.yaml`.
- The pipeline will automatically use gTTS as a fallback.

---

## Step 3: Run the pipeline

```bash
python scripts/inference/run_pipeline.py --input episode.mp4 --lang xx
```

The pipeline will:
1. Extract and transcribe audio (Japanese ASR)
2. Translate to the target language via Google Translate
3. Apply dialect post-processing if `dialect_post_process: true`
4. Synthesize speech using the configured TTS engine
5. Mix and align the final audio

If the Coqui model is not installed, the pipeline falls back to gTTS and continues gracefully.

---

## Step 4: (Optional) Add a dialect post-processor

If the language needs rule-based dialect transformation after translation:

1. Create `scripts/dialects/xx_dialect.py` following the pattern in `scripts/dialects/hindi_to_haryanvi.py`.
   - Accept `--input` and `--output` argparse arguments.
   - Read a JSON list of segment dicts, add a `text_dialect` field, write back to output.
2. Set in `languages.yaml`:
   ```yaml
   dialect_post_process: true
   dialect_script: scripts/dialects/xx_dialect.py
   ```

The pipeline will automatically invoke the script as Stage 4b.

---

## Example: Bengali (already configured)

Bengali is already present in `configs/languages.yaml` with `status: planned`.
Running the pipeline with `--lang bn` will:
- Translate Japanese → Bengali via Google Translate
- Attempt to load `tts_models/bn/cv/vits` (Coqui)
- Fall back to gTTS Bengali if the Coqui model is not installed

No code changes were required to add Bengali support.
