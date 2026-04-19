# Spec: Phase 1 — Core Pipeline (Hindi)

## Overview
Get the full 6-stage pipeline running end-to-end on a real anime clip and produce a Hindi dubbed audio file. All scripts already exist — this phase is about wiring, testing, and fixing bugs.

---

## Requirements

1. All Python dependencies install cleanly from `requirements.txt`
2. `.env` file is configured with `HF_TOKEN` for pyannote diarization
3. `extract_audio.py` produces a clean 22050 Hz mono WAV from any `.mp4` input
4. `diarize_speakers.py` identifies 2+ speakers with correct time segments
5. `asr_transcribe.py` transcribes Japanese with word-level timestamps, auto-detects GPU/CPU
6. `translate.py` translates all segments to Hindi, preserving timing metadata
7. `tts_hindi.py` synthesizes Hindi audio for every segment, with CPU/GPU auto-detection
8. `align_and_mix.py` assembles all TTS segments into a single WAV matching source duration
9. `run_pipeline.py` orchestrates all 6 stages with `--start-stage` resume support
10. Translation output includes post-processed `text_cleaned` field alongside raw `text_translated`
11. TTS output manifest includes `stretch_ratio` field per segment for monitoring

---

## Tasks

### Task 1: Install Dependencies
Install all packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

If `torch` fails due to CUDA version mismatch, visit https://pytorch.org/get-started/locally/ and install the correct build. Then re-run the rest of `requirements.txt`.

**Done when:** `python -c "import torch, TTS, faster_whisper, pyannote.audio; print('OK')"` prints `OK`.

---

### Task 2: Configure Environment
Set up `.env` with HuggingFace token.

```bash
cp .env.example .env
```

1. Create a HuggingFace account and generate a token at https://huggingface.co/settings/tokens
2. Accept the pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Add `HF_TOKEN=hf_your_token_here` to `.env`

**Done when:** `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ['HF_TOKEN'])"` prints your token without error.

---

### Task 3: Test Audio Extraction
Verify `scripts/preprocessing/extract_audio.py` works on a sample clip.

```bash
python scripts/preprocessing/extract_audio.py \
  --input sample.mp4 \
  --output data/processed/audio.wav
```

Check: file exists, plays correctly, is 22050 Hz mono. Use `librosa.get_duration()` to confirm duration matches source.

**Done when:** Clean WAV produced with no ffmpeg errors and correct properties.

---

### Task 4: Test Speaker Diarization
Run `scripts/preprocessing/diarize_speakers.py` and verify speaker segments.

```bash
python scripts/preprocessing/diarize_speakers.py \
  --audio data/processed/audio.wav \
  --output data/processed/diarization.json
```

Verify `diarization.json` contains entries like:
```json
[{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00", "duration": 2.7}]
```

**Done when:** JSON has 2+ distinct `SPEAKER_*` labels with non-overlapping time ranges.

---

### Task 5: Test Japanese ASR
Run `scripts/preprocessing/asr_transcribe.py` and verify Japanese transcript.

```bash
python scripts/preprocessing/asr_transcribe.py \
  --audio data/processed/audio.wav \
  --model medium \
  --output data/processed/transcript_ja.json
```

Use `medium` model for speed testing. Switch to `large-v3` for production.

**Done when:** `transcript_ja.json` contains Japanese text with `start`, `end`, and `words` (word-level timestamps) per segment.

---

### Task 6: Fix ASR GPU/CPU Auto-Detection
Edit `scripts/preprocessing/asr_transcribe.py` — the current script hardcodes `device="cuda"`. Fix it.

Find the model initialization (around line 13) and replace with:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel(model_size, device=device, compute_type=compute_type)
```

Also add a `--device` CLI argument for manual override. Print which device is being used at startup.

**Done when:** Script runs on both a GPU machine and a CPU-only machine without manual config edits.

---

### Task 7: Test Translation
Run `scripts/preprocessing/translate.py` to translate Japanese transcript to Hindi.

```bash
python scripts/preprocessing/translate.py \
  --input data/processed/transcript_ja.json \
  --output data/processed/transcript_hi.json
```

**Done when:** Every segment in `transcript_hi.json` has `text_original` (Japanese) and `text_translated` (Hindi) fields. Manually spot-check 5–10 lines.

---

### Task 8: Add Hindi Text Cleaning to Translate Script
Edit `scripts/preprocessing/translate.py` to add post-processing after Google Translate.

Add this function:

```python
def clean_hindi_text(text: str) -> str:
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    anime_term_map = {
        "सेनपाई": "सेनपाई",
        "नारुतो": "नारुतो",
        "शिनोबी": "शिनोबी",
    }
    for k, v in anime_term_map.items():
        text = text.replace(k, v)
    return text
```

Call it after translation and save the result as `text_cleaned` alongside `text_translated` in each segment dict.

**Done when:** Output JSON has both `text_translated` and `text_cleaned` fields per segment.

---

### Task 9: Test Hindi TTS
Run `scripts/inference/tts_hindi.py`. The first run downloads the Coqui Hindi VITS model (~150 MB automatically).

```bash
python scripts/inference/tts_hindi.py \
  --input data/processed/transcript_hi.json \
  --output-dir data/tts_output/
```

If CPU-only: find `.to("cuda")` in `synthesize_hindi()` and change it to `.to("cpu")`.

**Done when:** Files `data/tts_output/seg_0000.wav` through `seg_NNNN.wav` exist and are intelligible Hindi audio.

---

### Task 10: Add TTS Stretch Ratio Logging
Edit `scripts/inference/tts_hindi.py`. After generating each segment's audio, compute and log the stretch ratio.

Add inside the synthesis loop after writing `out_file`:

```python
import librosa as _lb
tts_audio, sr_ = _lb.load(out_file, sr=22050)
tts_dur = len(tts_audio) / sr_
orig_dur = seg['end'] - seg['start']
seg['tts_duration'] = round(tts_dur, 3)
seg['original_duration'] = round(orig_dur, 3)
seg['stretch_ratio'] = round(tts_dur / orig_dur, 3) if orig_dur > 0 else 1.0
```

**Done when:** `data/tts_output/segments.json` has `stretch_ratio` per segment. Most values should be between 0.7 and 1.5.

---

### Task 11: Test Alignment and Final Mix
Get the source audio duration, then run `scripts/inference/align_and_mix.py`.

```python
# Get duration first:
import librosa
print(librosa.get_duration(path="data/processed/audio.wav"))
```

```bash
python scripts/inference/align_and_mix.py \
  --segments data/tts_output/segments.json \
  --output outputs/final_hi_dub.wav \
  --duration <total_seconds_from_above>
```

**Done when:** `outputs/final_hi_dub.wav` plays without silence gaps, clipping, or crashes. Duration is within 5% of source.

---

### Task 12: Run Full End-to-End Pipeline
Run the master pipeline orchestrator on a 2-minute anime clip.

```bash
python scripts/inference/run_pipeline.py --input sample.mp4 --lang hi
```

Useful flags:
- `--start-stage 3` — resume from Stage 3 (skip re-extraction)
- `--skip-diarize` — skip speaker diarization for faster iteration
- `--model-size medium` — use smaller Whisper for speed testing

**Done when:** All 6 stages complete, `outputs/final_hi_dub.wav` exists and plays back Hindi dialogue with correct timing.

---

## Acceptance Criteria

- [-] `python -c "import torch, TTS, faster_whisper, pyannote.audio"` — no errors
- [ ] `data/processed/audio.wav` — 22050 Hz, mono, matches source duration
- [ ] `data/processed/diarization.json` — 2+ speakers, valid timestamps
- [ ] `data/processed/transcript_ja.json` — Japanese text, word-level timestamps, >90% dialogue captured
- [ ] `data/processed/transcript_hi.json` — `text_translated` + `text_cleaned` fields present
- [ ] `data/tts_output/segments.json` — `stretch_ratio` field per segment
- [ ] `outputs/final_hi_dub.wav` — plays Hindi audio, no crashes, no clipping
- [ ] `run_pipeline.py` completes end-to-end without manual intervention
- [ ] ASR script runs on both GPU and CPU without manual edits
