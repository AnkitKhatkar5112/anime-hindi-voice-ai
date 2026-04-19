# Spec: Phase 2 — Quality Improvements

## Overview
Improve the naturalness and expressiveness of the Hindi dub through voice cloning, emotion detection, prosody transfer, and better timing alignment. Phase 1 must be complete before starting this phase.

**Prerequisite:** `outputs/final_hi_dub.wav` exists from Phase 1.

---

## Requirements

1. Each detected speaker gets a unique voice embedding stored as a `.npy` file
2. TTS uses per-speaker embeddings so different characters sound different
3. Every segment in the transcript JSON has `emotion` and `emotion_intensity` fields
4. Segments with `emotion_intensity > 0.7` are synthesized with Bark TTS for expressiveness
5. Prosody transfer applies the Japanese pitch contour shape to the Hindi TTS output
6. DTW-based alignment replaces uniform time-stretch for more natural timing
7. A valid `.srt` subtitle file is generated alongside the dubbed audio

---

## Tasks

### Task 1: Extract Speaker Voice Embeddings
Run the already-implemented embedding extractor.

```bash
python scripts/training/extract_voice_embeddings.py \
  --audio data/processed/audio.wav \
  --diarization data/processed/diarization.json \
  --output-dir data/voice_references/embeddings/
```

Check that files like `data/voice_references/embeddings/SPEAKER_00.npy` exist.  
Check `speaker_manifest.json` — each speaker needs `total_audio_seconds > 3.0`.

**Done when:** Every detected speaker has a `.npy` embedding file.

---

### Task 2: Apply Speaker Embeddings in TTS
Edit `scripts/inference/tts_hindi.py` — update `synthesize_hindi()` to accept and use speaker embeddings.

Changes:
1. Add `diarization_json: str = None` parameter to `synthesize_hindi()`
2. Load diarization data at function start
3. For each segment, find the overlapping speaker by comparing `seg['start']` / `seg['end']` against diarization timestamps
4. Load the matching `data/voice_references/embeddings/{speaker_id}.npy`
5. Pass it as `speaker_embedding` (numpy array) or `speaker_wav` path to Coqui TTS (check your model's API)
6. Log `speaker_id` used per segment into the output manifest

**Done when:** Listening to `outputs/final_hi_dub.wav` — different characters have noticeably different voice qualities.

---

### Task 3: Create Emotion Detection Script
Create `scripts/preprocessing/detect_emotion.py`. This script adds `emotion` and `emotion_intensity` to each segment.

Use a text-based approach (faster, no extra audio model needed):

```python
from transformers import pipeline

def detect_emotion(segments_json: str, output_json: str):
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    # For each segment:
    #   run classifier on text_translated (translate to English first if needed,
    #   or use a multilingual model)
    #   pick top emotion and its score as emotion_intensity
    #   map to one of: neutral, happy, sad, angry, surprised, fearful
    #   add {"emotion": "angry", "emotion_intensity": 0.85} to segment
```

The script should accept `--input` and `--output` argparse arguments and save the enriched JSON.

**Done when:** Every segment in the output JSON has `emotion` (string) and `emotion_intensity` (0.0–1.0 float) fields.

---

### Task 4: Add Bark TTS for High-Emotion Segments
Edit `scripts/inference/tts_hindi.py` to use Bark for expressive lines.

First install:
```bash
pip install git+https://github.com/suno-ai/bark.git
```

In `synthesize_hindi()`, add a branch before the Coqui call:

```python
from bark import generate_audio, SAMPLE_RATE
import soundfile as sf

BARK_VOICE_MAP = {
    "angry":     "v2/hi_speaker_3",
    "happy":     "v2/hi_speaker_1",
    "sad":       "v2/hi_speaker_5",
    "surprised": "v2/hi_speaker_2",
    "fearful":   "v2/hi_speaker_4",
    "neutral":   "v2/hi_speaker_0",
}

if seg.get("emotion_intensity", 0) > 0.7:
    prompt = BARK_VOICE_MAP.get(seg.get("emotion", "neutral"), "v2/hi_speaker_0")
    audio_array = generate_audio(text, history_prompt=prompt)
    sf.write(out_file, audio_array, SAMPLE_RATE)
    seg["tts_engine"] = "bark"
else:
    tts.tts_to_file(text=text, file_path=out_file, ...)
    seg["tts_engine"] = "coqui"
```

**Note:** Bark is ~5× slower than Coqui. Only use it for the high-emotion branch.

**Done when:** Segments with high emotion intensity (shouting, crying, laughing) sound more expressive than the Coqui baseline. `tts_engine` field is logged in the manifest.

---

### Task 5: Create Prosody Transfer Script
Create `scripts/inference/prosody_transfer.py`. This transfers the pitch contour *shape* from the original Japanese segment onto the Hindi TTS output.

```python
import pyworld as pw
import numpy as np
import librosa
import soundfile as sf

def transfer_prosody(original_wav: str, tts_wav: str, output_wav: str, sr: int = 22050):
    orig, _ = librosa.load(original_wav, sr=sr)
    tts, _  = librosa.load(tts_wav, sr=sr)

    orig_f0, orig_sp, orig_ap = pw.wav2world(orig.astype(np.float64), sr)
    tts_f0,  tts_sp,  tts_ap  = pw.wav2world(tts.astype(np.float64), sr)

    # 1. Normalize both F0 curves (remove speaker-specific mean/std)
    # 2. Resample orig_f0 shape to match length of tts_f0
    # 3. Scale to tts speaker's mean/std
    # 4. Synthesize with transferred F0

    out = pw.synthesize(transferred_f0, tts_sp, tts_ap, float(sr))
    sf.write(output_wav, out.astype(np.float32), sr)
```

Wire it into `align_and_mix.py` as an optional `--prosody-transfer` flag that runs after time-stretching each segment.

**Done when:** Hindi audio follows the emotional pitch arc of the original Japanese (rises/falls in the right places). PESQ score should not drop compared to non-transferred baseline.

---

### Task 6: Upgrade to DTW-Based Alignment
Edit `scripts/inference/align_and_mix.py`. Replace the uniform time-stretch in `time_stretch_segment()` with DTW-based warping.

```python
import librosa
import numpy as np

def dtw_align(orig_audio: np.ndarray, tts_audio: np.ndarray, sr: int) -> np.ndarray:
    orig_mfcc = librosa.feature.mfcc(y=orig_audio, sr=sr, n_mfcc=13)
    tts_mfcc  = librosa.feature.mfcc(y=tts_audio,  sr=sr, n_mfcc=13)
    _, wp = librosa.sequence.dtw(orig_mfcc, tts_mfcc, subseq=True)
    # wp is the warping path: use it to resample tts_audio frames
    # Map frame indices back to sample indices and interpolate
    tts_frames = wp[:, 1]
    orig_frames = wp[:, 0]
    # Reconstruct tts_audio resampled along the warping path
    ...
    return warped_audio
```

Keep the old uniform stretch as fallback if DTW produces artifacts (stretch ratio > 1.5 or < 0.6).

**Done when:** Aligned audio sounds more natural than uniform time-stretch. Fewer "chipmunk" or "slow-motion" artifacts on fast/slow segments.

---

### Task 7: Generate SRT Subtitles
Run the already-implemented subtitle generator.

```bash
python scripts/inference/generate_subtitles.py \
  --input data/processed/transcript_hi.json \
  --output outputs/subtitles_hi.srt
```

Open `outputs/subtitles_hi.srt` in VLC (Subtitles → Add Subtitle File). Confirm Hindi text appears at the correct timestamps.

**Done when:** Valid `.srt` file — correct format (index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text, blank line), subtitles visible in VLC at right times.

---

## Acceptance Criteria

- [ ] `data/voice_references/embeddings/SPEAKER_*.npy` — exists for every detected speaker
- [ ] `speaker_manifest.json` — `total_audio_seconds > 3.0` per speaker
- [ ] TTS output — listening reveals distinct voice characteristics per character
- [ ] `transcript_hi.json` (after emotion pass) — `emotion` and `emotion_intensity` on every segment
- [ ] `tts_output/segments.json` — `tts_engine` field is `"bark"` or `"coqui"` per segment
- [ ] High-emotion lines — noticeably more expressive than Coqui baseline when played back
- [ ] `prosody_transfer.py` exists and runs on a single segment pair without error
- [ ] PESQ score after DTW alignment ≥ PESQ score from Phase 1 uniform stretch
- [ ] `outputs/subtitles_hi.srt` — valid SRT, renders in VLC at correct times
