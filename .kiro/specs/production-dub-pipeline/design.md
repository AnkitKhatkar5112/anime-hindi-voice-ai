# Technical Design Document: Production Dub Pipeline

## 1. Overview

This document describes the technical architecture for the production-level anime dubbing pipeline. The system takes a video file (and optional `.srt` subtitle file) as input and produces a multi-track dubbed video with Hindi audio, BGM/SFX preservation, per-character voice consistency, emotion-driven prosody, and LUFS-normalized audio output.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pipeline Orchestrator                        │
│                      scripts/inference/run_pipeline.py               │
│  --fast-mode  --resume-from N  --force-restart  --clear-cache        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │              Stage 1: Ingest             │
          │  extract_audio.py  │  SRT_Parser         │
          │  (FFmpeg)          │  (pysrt / custom)   │
          └────────────────────┬────────────────────┘
                               │  raw WAV + Segment list
          ┌────────────────────▼────────────────────┐
          │         Stage 2: Source Separation       │
          │         Source_Separator (Demucs)        │
          │   vocals stem  │  background stem        │
          │   SNR check → fallback to ducking        │
          └────────────────────┬────────────────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │      Stage 3: Speaker Diarization        │
          │  --fast-mode: Resemblyzer ONLY (~20-30s) │
          │  standard:    pyannote (~2-5 min)        │
          │  Voice_Embedding_Clusterer               │
          └────────────────────┬────────────────────┘
                               │  speaker IDs per segment
          ┌────────────────────▼────────────────────┐
          │       Stage 4: Translation & Adaptation  │
          │  Helsinki-NLP/opus-mt-ja-hi (primary)    │
          │  DeepL / Google Translate (fallback)     │
          │  Dubbing_Adapter: GPT-4o → Ollama →      │
          │    rule-based (honorific + trim)         │
          │  Backend locked for entire episode       │
          └────────────────────┬────────────────────┘
                               │  localized Hindi text per segment
          ┌────────────────────▼────────────────────┐
          │         Stage 5: Emotion Detection       │
          │  Emotion_Engine                          │
          │  rule-based (text patterns) →            │
          │  AI enrichment (transformers)            │
          └────────────────────┬────────────────────┘
                               │  emotion + intensity per segment
          ┌────────────────────▼────────────────────┐
          │      Stage 6: Voice Assignment           │
          │  Character_Voice_Mapper                  │
          │  configs/character_voices.yaml           │
          │  data/voice_maps/{episode_id}_voice_map  │
          │  (includes tts_backend key)              │
          └────────────────────┬────────────────────┘
                               │  VoiceProfile per segment
          ┌────────────────────▼────────────────────┐
          │         Stage 7: TTS Synthesis           │
          │  TTS backend locked per episode          │
          │  (NeMo → Coqui VITS → gTTS)             │
          │  Emotion_Engine prosody application      │
          │  Cache: data/cache/tts/                  │
          └────────────────────┬────────────────────┘
                               │  per-segment WAV files
          ┌────────────────────▼────────────────────┐
          │    Stage 8: Overlap Resolution + Mixing  │
          │  Overlap < 200ms: trim earlier segment   │
          │  Overlap ≥ 200ms: blend at 40% volume   │
          │  (no segments discarded)                 │
          │  Mixer (pyloudnorm LUFS + peak ceiling)  │
          │  background stem / ducking blend         │
          └────────────────────┬────────────────────┘
                               │  final_hindi_dubbed.wav
          ┌────────────────────▼────────────────────┐
          │    Stage 8b: Subtitle Processing         │
          │  Subtitle_Classifier + Subtitle_Splitter │
          │  Sign_Translator (literal, no Adapter)   │
          │  OCR_Engine (optional, --ocr-signs)      │
          │    OCR timestamps aligned to ±500ms SRT  │
          │  Subtitle_Writer (SRT / ASS output)      │
          └────────────────────┬────────────────────┘
                               │  outputs/{episode_id}.hi.srt/.ass
          ┌────────────────────▼────────────────────┐
          │           Stage 9: Muxing               │
          │  Muxer (FFmpeg subprocess)               │
          │  2-track output: jpn + hin               │
          └────────────────────┬────────────────────┘
                               │  output.mp4 / output.mkv
          ┌────────────────────▼────────────────────┐
          │        Stage 10: Output Validation       │
          │  Validator (FFprobe subprocess)          │
          │  logs/output_validation_{id}.json        │
          └─────────────────────────────────────────┘
```

---

## 3. Pipeline Stages: Inputs, Outputs, and Responsibilities

### Stage 1: Ingest

| Item | Detail |
|------|--------|
| Script | `scripts/preprocessing/extract_audio.py`, `scripts/preprocessing/srt_parser.py` (new) |
| Inputs | `{episode}.mp4/.mkv`, optional `{episode}.srt`, `--subtitle-offset` (ms) |
| Outputs | `data/raw_audio/extracted.wav`, `List[Segment]` |

**SRT_Parser behavior:**
- Reads `.srt` with `pysrt` or custom parser; handles UTF-8 and UTF-8-BOM.
- Applies `--subtitle-offset` (clamped to ≥ 0ms) to all `start`/`end` times.
- Logs offset value and segment count adjusted.
- If zero valid entries → returns error, pipeline falls back to Whisper ASR.

### Stage 2: Source Separation

| Item | Detail |
|------|--------|
| Script | `scripts/preprocessing/separate_audio.py` (new) |
| Inputs | `data/raw_audio/extracted.wav` |
| Outputs | `data/processed/vocals.wav`, `data/processed/background.wav`, `data/processed/separation_quality.json` |

**SNR quality check:**
```python
snr_db = 10 * log10(signal_power / noise_power)  # computed on vocals stem
if snr_db < 10.0:
    trigger_ducking_fallback()
```
- Fallback: full original audio used as background; volume reduced to 40% during dialogue segments, restored to 100% outside.
- `separation_quality.json` schema: `{"snr_db": float, "fallback_triggered": bool, "model": str}`.

### Stage 3: Speaker Diarization

| Item | Detail |
|------|--------|
| Script | `scripts/preprocessing/diarize_speakers.py` (existing, extended) |
| Inputs | `data/processed/vocals.wav` (or `extracted.wav` if fallback), `--fast-mode` flag |
| Outputs | `data/processed/diarization.json` with speaker IDs per time range |

**Fast-mode flag (`--fast-mode`):**
- WHEN `--fast-mode` is set: skip pyannote diarization entirely; use `Voice_Embedding_Clusterer` (Resemblyzer) as the **only** speaker identification method, regardless of whether an SRT file is provided. Reduces Stage 3 from ~2–5 min to ~20–30 sec on CPU.
- WHEN `--fast-mode` is NOT set: use pyannote as primary (existing behavior).
- The active diarization mode SHALL be logged at pipeline startup: `"Diarization mode: fast (Resemblyzer)"` or `"Diarization mode: standard (pyannote)"`.

**SRT path (no embedded speaker names):** `Voice_Embedding_Clusterer` extracts Resemblyzer embeddings for each SRT-timed segment from the vocals stem, clusters with agglomerative clustering (cosine distance threshold 0.25), and assigns `SPEAKER_XX` IDs.

### Stage 4: Translation & Adaptation

| Item | Detail |
|------|--------|
| Script | `scripts/preprocessing/translate.py` (existing), `scripts/preprocessing/dubbing_adapter.py` (new) |
| Inputs | `List[Segment]` with source text |
| Outputs | `List[Segment]` with `translated_text` and `dubbed_text` |

**Primary translation stack (locked per episode):**
- Translation model: `Helsinki-NLP/opus-mt-ja-hi` (primary, offline-capable)
- Fallback: DeepL API → Google Translate API (only if Helsinki-NLP fails)
- The selected translation backend is locked for the entire episode and logged at startup: `"Translation backend: Helsinki-NLP/opus-mt-ja-hi"` (or fallback name)

**Dubbing_Adapter backend priority:**
1. GPT-4o — if `OPENAI_API_KEY` environment variable is set
2. Local LLM via Ollama — if Ollama is running and reachable
3. Rule-based post-processing — honorific substitution + length trimming (final fallback, always available)

**Dubbing_Adapter** enforces:
- Conversational Hindi phrasing
- Anime honorific localization
- Character count ≤ `duration_s × 14 × 1.3`
- No untranslated Japanese characters

### Stage 5: Emotion Detection

| Item | Detail |
|------|--------|
| Script | `scripts/preprocessing/detect_emotion.py` (extended) |
| Inputs | `List[Segment]` with `dubbed_text` |
| Outputs | `List[Segment]` with `emotion: str`, `emotion_intensity: float` |

**Two-stage strategy:**
1. Rule-based: scan text for `!`, `?`, Hindi keywords (`nahi!`, `kyu!`, `bachao`, `ruko`, `dekho`). Maps patterns to emotion labels and baseline intensity.
2. AI enrichment: `transformers` pipeline (`j-hartmann/emotion-english-distilroberta-base` or Hindi equivalent). Overwrites rule-based result only if confidence > 0.7.
3. Fallback: if AI fails, rule-based result is kept. Every segment always gets a valid `emotion` from `{neutral, happy, angry, sad, excited, fearful}`.

### Stage 6: Voice Assignment

| Item | Detail |
|------|--------|
| Script | `scripts/inference/character_voice_mapper.py` (new) |
| Inputs | `List[Segment]` with speaker IDs, `configs/character_voices.yaml` |
| Outputs | `data/voice_maps/{episode_id}_voice_map.json`, `List[Segment]` with `voice_profile` |

**Assignment logic:**
- Load existing map from `data/voice_maps/{episode_id}_voice_map.json` if present (supports restarts).
- For each new speaker ID: pick the next unassigned profile from `character_voices.yaml`.
- If profiles exhausted: cycle with warning log.
- `--shared-voice-map` flag: load a cross-episode map and extend it.

### Stage 7: TTS Synthesis

| Item | Detail |
|------|--------|
| Script | `scripts/inference/tts_hindi.py` (extended) |
| Inputs | `List[Segment]` with `dubbed_text`, `voice_profile`, `emotion`, `emotion_intensity` |
| Outputs | `data/tts_output/{segment_id}.wav` per segment |

**TTS backend locking per episode:**
- The TTS backend is selected **once** at pipeline startup and locked for the entire episode.
- The selected backend is written to `data/voice_maps/{episode_id}_voice_map.json` under a `tts_backend` key.
- IF a voice map already exists for the episode with a `tts_backend` key, the pipeline SHALL reuse that backend (not re-detect) to ensure consistency on resume.
- This prevents voice quality shifts mid-episode when resuming from a checkpoint.

**Backend selection (auto-detect at startup, only if no existing voice map):**
```
1. NeMo FastPitch+HiFiGAN  → if CUDA available AND nemo_toolkit installed AND Hindi supported
2. Coqui VITS Hindi model  → if tts library installed AND Hindi model available
3. gTTS                    → CPU-only fallback, always available
```

**Prosody application (Emotion_Engine):**

| Emotion | Rate modifier | Pitch shift |
|---------|--------------|-------------|
| angry | +10–20% | +2 semitones |
| sad | -10–15% | -1 semitone |
| happy/excited | +5–15% | +1 semitone |
| neutral | 0% | 0 |

- `emotion_intensity > 0.7` → route to Bark TTS with emotion-matched Hindi preset.
- Time-stretch via `librosa.effects.time_stretch` (ratio ≤ 1.4×) if TTS output exceeds segment duration.
- If stretch insufficient: shorten text and re-synthesize (max 1 retry).

### Stage 8: Overlap Resolution + Mixing

| Item | Detail |
|------|--------|
| Script | `scripts/inference/align_and_mix.py` (extended) |
| Inputs | `data/tts_output/*.wav`, `data/processed/background.wav`, `List[Segment]` |
| Outputs | `data/processed/final_hindi_dubbed.wav`, `logs/mixing_report_{episode_id}.json` |

**Overlap resolution (pre-mix):**
```
for each consecutive pair (seg_i, seg_j) where seg_j.start < seg_i.end:
    overlap_ms = (seg_i.end - seg_j.start) * 1000
    if overlap_ms < 200:
        seg_i.end = seg_j.start          # trim earlier segment
    else:
        # BLEND: reduce lower-priority segment volume to 40% during overlap window
        lower_priority = seg_j if seg_i.emotion_intensity >= seg_j.emotion_intensity else seg_i
        lower_priority.overlap_blend = True   # flag for mixer
        lower_priority.overlap_gain = 0.40    # 40% volume during overlap window
        # both segments are preserved — no segment is discarded
```

**Blend behavior:**
- WHEN overlap >= 200ms: the lower-priority segment (lower `emotion_intensity`) is volume-reduced to 40% of its original level during the overlap window, then both segments are blended.
- WHEN overlap < 200ms: trim the earlier segment's end time to the later segment's start time (no blending needed).
- The mixing report SHALL record `"blended": true` for segments that were volume-reduced during overlap.
- No segment SHALL be discarded due to overlap — all speech is preserved.

**LUFS mixing pipeline:**
```
1. Place each TTS segment on timeline with 10ms fade-in/fade-out
2. Sum all TTS segments into dialogue_track (float32)
3. Blend: mixed = background * 0.20 + dialogue_track * 0.85
   (ducking mode: background * 0.40 outside dialogue, * 0.16 during dialogue)
4. Measure integrated LUFS with pyloudnorm.Meter(rate=22050)
5. Log pre-normalization LUFS
6. Apply LUFS gain to reach -16 LUFS
7. Apply peak ceiling: clip to -1.0 dBFS
8. Log post-normalization LUFS
9. Write 22050 Hz, 16-bit PCM WAV
```

**Mixing report JSON schema:**
```json
{
  "episode_id": "ep01",
  "pre_normalization_lufs": -23.4,
  "post_normalization_lufs": -16.0,
  "overlaps_detected": 3,
  "overlaps_resolved": 3,
  "segments": [
    {
      "segment_id": "seg_001",
      "start": 1.2,
      "end": 3.8,
      "speaker_id": "SPEAKER_00",
      "stretch_ratio": 1.0,
      "gain_db": 2.1,
      "discarded": false,
      "blended": false
    }
  ]
}
```

### Stage 8b: Subtitle Processing Module

| Item | Detail |
|------|--------|
| Script | `scripts/inference/subtitle_processor.py` (new), `scripts/inference/ocr_engine.py` (new) |
| Inputs | `List[Segment]` with `dubbed_text`, `start`, `end`, `speaker_id`; original video file (OCR fallback); `--subtitle-format` flag (`srt`\|`ass`); `--ocr-signs` flag |
| Outputs | `outputs/{episode_id}.hi.srt` or `outputs/{episode_id}.hi.ass`; `data/processed/ocr_signs.json` (if OCR triggered) |

**Position in pipeline:** After Stage 8 (Overlap Resolution + Mixing), before Stage 9 (Muxing).

**Component diagram:**
```
List[Segment]
│
├── Subtitle_Classifier
│   ├── sign patterns (ALL CAPS, brackets, sound effects)
│   ├── dialogue patterns (sentence structure, mixed case)
│   └── mixed → Subtitle_Splitter → [sign_seg, dialogue_seg]
│
├── dialogue segments → existing dubbing pipeline (already processed)
│
├── sign segments → Sign_Translator (literal translation, no Dubbing_Adapter)
│
├── OCR_Engine (optional, --ocr-signs flag)
│   ├── FFmpeg frame extraction at 1 fps
│   ├── easyocr (Japanese model)
│   ├── filter: keep top 80% of frame only
│   └── → additional sign segments
│
└── Subtitle_Writer
    ├── SRT mode: sequential index, HH:MM:SS,mmm timecodes, UTF-8
    │   └── sign text wrapped in [...] brackets
    └── ASS mode:
        ├── sign subtitles: \an8 (top alignment)
        ├── dialogue subtitles: default (bottom)
        └── Style: Noto Sans Devanagari font
```

**Classification rules (code-level):**
```python
import re

SIGN_PATTERNS = [
    r'^\[.*\]$',           # fully bracketed: [BANG]
    r'^\(.*\)$',           # fully parenthesized: (sound effect)
    r'^[A-Z\s\d\W]+$',    # ALL CAPS
    r'^\w{1,2}$',          # single/double word, no sentence
]

SOUND_EFFECTS = {'bang', 'crash', 'music', 'sfx', 'sound', 'noise'}

def classify(text: str) -> str:
    stripped = text.strip()
    # Check for mixed: has both bracketed and non-bracketed content
    if re.search(r'\[.*?\]', stripped) and re.sub(r'\[.*?\]', '', stripped).strip():
        return 'mixed'
    for pattern in SIGN_PATTERNS:
        if re.match(pattern, stripped):
            return 'sign'
    words = stripped.lower().split()
    if len(words) <= 2 and all(w in SOUND_EFFECTS for w in words):
        return 'sign'
    return 'dialogue'
```

**Mixed splitting:**
```python
def split_mixed(segment: Segment) -> tuple[Segment, Segment]:
    sign_text = ' '.join(re.findall(r'\[.*?\]', segment.source_text))
    dialogue_text = re.sub(r'\[.*?\]', '', segment.source_text).strip()
    sign_seg = replace(segment, source_text=sign_text, subtitle_type='sign')
    dialogue_seg = replace(segment, source_text=dialogue_text, subtitle_type='dialogue')
    return sign_seg, dialogue_seg
```

**ASS style block:**
```
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Alignment
Style: Dialogue,Noto Sans Devanagari,48,&H00FFFFFF,0,2
Style: Sign,Noto Sans Devanagari,40,&H00FFFF00,1,8
```

**OCR fallback pipeline:**
```python
# 1. Extract frames
# ffmpeg -i {video} -vf fps=1 data/processed/frames/frame_%04d.jpg

# 2. Run OCR
import easyocr
reader = easyocr.Reader(['ja'])
for frame_path in frame_paths:
    results = reader.readtext(frame_path)
    # filter: keep only detections in top 80% of frame height
    signs = [r for r in results if r[0][0][1] < frame_height * 0.80]

# 3. Align OCR timestamp to nearest SRT/ASR segment boundary within ±500ms
def align_ocr_timestamp(raw_ts: float, segment_boundaries: list[float]) -> float:
    candidates = [b for b in segment_boundaries if abs(b - raw_ts) <= 0.5]
    if candidates:
        aligned = min(candidates, key=lambda b: abs(b - raw_ts))
        log(f"[OCR ALIGN] {raw_ts:.3f}s → {aligned:.3f}s (delta {aligned-raw_ts:+.3f}s)")
        return aligned
    log(f"[OCR RAW] {raw_ts:.3f}s (no boundary within ±500ms)")
    return raw_ts

# 4. Translate and add to subtitle list
aligned_count, raw_count = 0, 0
for sign in signs:
    start_ts = align_ocr_timestamp(sign.frame_timestamp, srt_boundaries)
    if start_ts != sign.frame_timestamp:
        aligned_count += 1
    else:
        raw_count += 1
    translated = translate_literal(sign[1])
    segments.append(Segment(
        source_text=sign[1],
        dubbed_text=f"[{translated}]",
        start=start_ts,
        end=start_ts + 2.0,
        subtitle_type='sign'
    ))
log(f"[OCR] {aligned_count} signs aligned to SRT boundary, {raw_count} used raw timestamp")
```

**OCR timestamp alignment rules:**
- OCR-detected sign timestamps SHALL be aligned to the nearest existing SRT/ASR segment boundary within ±500ms.
- IF a matching boundary exists within ±500ms: use that segment's start time as the sign's start time.
- IF no matching boundary exists: use the raw OCR frame timestamp.
- The OCR_Engine SHALL log the count of signs aligned vs. signs using raw timestamps.

---

### Stage 9: Muxing

| Item | Detail |
|------|--------|
| Script | `scripts/inference/mux_output.py` (new) |
| Inputs | original video file, `data/processed/final_hindi_dubbed.wav` |
| Outputs | `outputs/{episode_id}_dubbed.mp4` or `.mkv` |

**FFmpeg command (mp4):**
```
ffmpeg -i {input_video} -i {hindi_wav} \
  -map 0:v -map 0:a -map 1:a \
  -c:v copy -c:a:0 copy -c:a:1 aac \
  -metadata:s:a:0 language=jpn -metadata:s:a:0 title="Japanese" \
  -metadata:s:a:1 language=hin -metadata:s:a:1 title="Hindi" \
  -disposition:a:0 default \
  {output_path}
```
For `.mkv`: replace `-c:a:1 aac` with `-c:a:1 libopus`.

### Stage 10: Output Validation

| Item | Detail |
|------|--------|
| Script | `scripts/inference/validate_output.py` (new) |
| Inputs | output video file path, `episode_id` |
| Outputs | `logs/output_validation_{episode_id}.json` |

**Validation checks (via FFprobe JSON output):**
1. Audio stream count == 2
2. `|video_duration - hindi_audio_duration| <= 2.0` seconds
3. Hindi audio stream `tags.language == "hin"`
4. `os.path.getsize(output_path) > 0`

Raises `PipelineValidationError` on any failure. Writes full FFprobe JSON + per-check pass/fail to `logs/output_validation_{episode_id}.json`.

---

## 4. Key Data Models

### Segment

```python
@dataclass
class Segment:
    segment_id: str           # e.g. "seg_001"
    start: float              # seconds
    end: float                # seconds
    speaker_id: str           # e.g. "SPEAKER_00"
    source_text: str          # original Japanese / source language text
    translated_text: str      # literal translation
    dubbed_text: str          # localized Hindi dubbing text
    emotion: str              # neutral | happy | angry | sad | excited | fearful
    emotion_intensity: float  # 0.0 – 1.0
    voice_profile: str        # key from character_voices.yaml
    tts_audio_path: str       # path to synthesized WAV
    stretch_ratio: float      # 1.0 = no stretch
    discarded: bool           # True if removed by overlap resolution
    blended: bool             # True if volume-reduced during overlap blend
    overlap_gain: float       # gain applied during overlap window (default 1.0)
    subtitle_type: str        # "dialogue" | "sign" | "mixed" (set by Subtitle_Classifier)
```

### VoiceProfile

```python
@dataclass
class VoiceProfile:
    profile_id: str           # e.g. "voice_male_deep"
    display_name: str
    tts_backend: str          # "nemo" | "coqui" | "gtts" | "bark"
    model_path: str           # path or model name
    speaker_id: str           # speaker embedding ID within model
    pitch_offset: float       # semitones, relative to model default
    rate_multiplier: float    # 1.0 = default
```

### VoiceMap

```python
# data/voice_maps/{episode_id}_voice_map.json
{
  "episode_id": "ep01",
  "series_id": "naruto",
  "created_at": "2025-01-01T00:00:00Z",
  "tts_backend": "coqui",
  "mappings": {
    "SPEAKER_00": "voice_male_deep",
    "SPEAKER_01": "voice_female_bright",
    "SPEAKER_02": "voice_male_young"
  }
}
```

### MixingReport

```python
# logs/mixing_report_{episode_id}.json
{
  "episode_id": str,
  "pre_normalization_lufs": float,
  "post_normalization_lufs": float,
  "peak_ceiling_dbfs": -1.0,
  "background_level": float,
  "dialogue_level": float,
  "ducking_mode": bool,
  "overlaps_detected": int,
  "overlaps_resolved": int,
  "segments": List[SegmentMixEntry]  # each entry includes "blended": bool
}
```

---

## 5. Multi-Voice Assignment Logic

```
Character_Voice_Mapper
│
├── load_voice_profiles()        ← configs/character_voices.yaml
├── load_existing_map()          ← data/voice_maps/{episode_id}_voice_map.json (if exists)
│
├── for each segment in episode:
│   ├── if speaker_id in map → use existing profile
│   └── if speaker_id NOT in map:
│       ├── pick next unassigned profile (round-robin if exhausted)
│       ├── log assignment
│       └── persist updated map to JSON
│
└── Voice_Embedding_Clusterer (SRT path only)
    ├── extract Resemblyzer embedding per SRT segment (from vocals stem)
    ├── agglomerative clustering (cosine distance, threshold=0.25)
    └── assign SPEAKER_XX IDs → feed into Character_Voice_Mapper
```

---

## 6. TTS Backend Selection Strategy

```python
def select_tts_backend(lang="hi") -> TTSBackend:
    # Priority 1: NeMo
    if torch.cuda.is_available() and is_installed("nemo_toolkit"):
        if nemo_supports_hindi():
            log("TTS backend: NeMo FastPitch+HiFiGAN")
            return NeMoBackend()

    # Priority 2: Coqui VITS
    if is_installed("TTS"):
        model = find_coqui_hindi_model()
        if model:
            log("TTS backend: Coqui VITS")
            return CoquiBackend(model)

    # Priority 3: gTTS fallback
    log("TTS backend: gTTS (CPU fallback)")
    return GTTSBackend()
```

---

## 7. Emotion Detection Two-Stage Strategy

```
Input: dubbed_text (Hindi)
│
├── Stage 1: Rule-based
│   ├── count "!" → excited/angry signal
│   ├── count "?" → curious/fearful signal
│   ├── keyword match: {"nahi": angry, "kyu": curious, "bachao": fearful,
│   │                   "ruko": urgent, "haha"/"wah": happy}
│   └── → (emotion_label, base_intensity)
│
└── Stage 2: AI enrichment (optional, transformers)
    ├── run emotion classifier on text
    ├── if confidence > 0.7 AND model available:
    │   └── override rule-based result
    └── if model fails:
        └── keep rule-based result (always valid)
```

---

## 8. Audio Source Separation + Fallback

```
extracted.wav
│
├── Demucs htdemucs model
│   ├── vocals.wav  → SNR check
│   │   ├── SNR >= 10 dB → use vocals for diarization, background for BGM
│   │   └── SNR < 10 dB  → FALLBACK: ducking mode
│   │       ├── background = full extracted.wav
│   │       ├── during dialogue segments: background * 0.40
│   │       └── outside dialogue: background * 1.00
│   └── background.wav → BGM input to Mixer
│
└── separation_quality.json
    {"snr_db": 14.2, "fallback_triggered": false, "model": "htdemucs"}
```

---

## 9. LUFS-Based Audio Mixing Pipeline

```python
import pyloudnorm as pyln
import numpy as np

def mix_and_normalize(segments, background, sample_rate=22050):
    # 1. Build dialogue track
    dialogue = np.zeros_like(background)
    for seg in segments:
        audio = load_wav(seg.tts_audio_path)
        audio = apply_fades(audio, fade_ms=10, sr=sample_rate)
        start_sample = int(seg.start * sample_rate)
        dialogue[start_sample:start_sample + len(audio)] += audio

    # 2. Blend
    mixed = background * 0.20 + dialogue * 0.85

    # 3. LUFS normalization
    meter = pyln.Meter(sample_rate)
    lufs_before = meter.integrated_loudness(mixed)
    log(f"Pre-normalization LUFS: {lufs_before:.1f}")

    normalized = pyln.normalize.loudness(mixed, lufs_before, -16.0)
    lufs_after = meter.integrated_loudness(normalized)
    log(f"Post-normalization LUFS: {lufs_after:.1f}")

    # 4. Peak ceiling at -1.0 dBFS
    peak = np.max(np.abs(normalized))
    ceiling = 10 ** (-1.0 / 20)
    if peak > ceiling:
        normalized = normalized * (ceiling / peak)

    return normalized
```

---

## 10. FFmpeg Multi-Track Muxing

The `Muxer` invokes FFmpeg as a subprocess. The exact command is logged to the pipeline log.

**MP4 output:**
```bash
ffmpeg -y \
  -i {input_video} \
  -i {hindi_wav} \
  -map 0:v -map 0:a -map 1:a \
  -c:v copy -c:a:0 copy -c:a:1 aac -b:a:1 192k \
  -metadata:s:a:0 language=jpn -metadata:s:a:0 title="Japanese" \
  -metadata:s:a:1 language=hin -metadata:s:a:1 title="Hindi" \
  -disposition:a:0 default \
  {output_path}
```

**MKV output:** replace `-c:a:1 aac -b:a:1 192k` with `-c:a:1 libopus -b:a:1 128k`.

Post-mux: verify `os.path.exists(output)` and `os.path.getsize(output) > 0`, raise on failure.

---

## 11. Output Validation with FFprobe

```python
import subprocess, json

def validate_output(output_path: str, episode_id: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", output_path],
        capture_output=True, text=True
    )
    probe = json.loads(result.stdout)

    checks = {}
    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]

    checks["audio_track_count"] = len(audio_streams) == 2
    checks["file_size_nonzero"] = os.path.getsize(output_path) > 0

    video_dur = float(video_streams[0].get("duration", 0))
    hindi_dur = float(audio_streams[1].get("duration", 0))
    checks["duration_within_2s"] = abs(video_dur - hindi_dur) <= 2.0

    hindi_lang = audio_streams[1].get("tags", {}).get("language", "")
    checks["hindi_language_tag"] = hindi_lang == "hin"

    report = {
        "episode_id": episode_id,
        "output_path": output_path,
        "checks": checks,
        "ffprobe_output": probe,
        "passed": all(checks.values())
    }

    with open(f"logs/output_validation_{episode_id}.json", "w") as f:
        json.dump(report, f, indent=2)

    if not report["passed"]:
        failures = [k for k, v in checks.items() if not v]
        raise PipelineValidationError(f"Output validation failed: {failures}")

    return report
```

---

## 12. Training Data Structure and Fine-Tuning Pipeline

```
training/
├── source/          # Japanese audio clips (WAV, 22050 Hz)
│   ├── ep01_seg001.wav
│   └── ...
├── dubbed/          # Hindi dubbed audio clips (WAV, 22050 Hz)
│   ├── ep01_seg001.wav
│   └── ...
├── wavs/            # LJ-Speech format symlinks / copies
│   └── ep01_seg001.wav
├── metadata.csv     # LJ-Speech: filename|text|normalized_text
└── dataset_stats.json
```

**Scripts:**
- `scripts/training/prepare_dataset.py`: reads `training/`, validates `metadata.csv` vs `wavs/`, produces LJ-Speech dataset.
- `scripts/training/finetune_vits.py`: fine-tunes Coqui VITS from `--base-model` checkpoint.
- `scripts/training/finetune_nemo.py` (new): fine-tunes NeMo FastPitch/VITS using same dataset.
- `scripts/training/dataset_stats.py` (new): reports total duration, speaker count, avg segment length, vocabulary coverage.

---

## 13. API and UI Changes

### FastAPI (`api/main.py`)

**New/modified endpoints:**

| Method | Path | Change |
|--------|------|--------|
| POST | `/dub` | Add optional `srt_file: UploadFile = None` and `subtitle_offset: int = 0` params |
| GET | `/download/{job_id}` | Return muxed video with correct `Content-Type` |
| GET | `/voices` | New: return `configs/character_voices.yaml` as JSON |

**File validation:** accept `.mp4`, `.mkv`, `.avi`; return HTTP 422 for others.

### Streamlit UI (`ui/app.py`)

- Add second `st.file_uploader` for optional `.srt` file.
- Add `st.number_input` for `--subtitle-offset` (ms, default 0, range -300 to +300).
- Show download button for muxed video on job completion.
- Poll `logs/current_stage.txt` for real-time stage display.

---

## 14. Folder Structure Additions

```
data/
├── processed/
│   ├── vocals.wav                        # Demucs vocals stem
│   ├── background.wav                    # Demucs background stem
│   ├── separation_quality.json           # SNR + fallback status
│   ├── final_hindi_dubbed.wav            # LUFS-normalized mixed audio
│   └── {episode_id}_checkpoint.json      # NEW: stage checkpoint for resume
├── cache/                                # NEW: pipeline cache directory
│   ├── translations/                     # JSON files keyed by SHA256
│   ├── embeddings/                       # .npy files keyed by SHA256
│   └── tts/                              # .wav files keyed by SHA256
├── voice_maps/
│   └── {episode_id}_voice_map.json       # speaker → voice profile mapping (+ tts_backend)
└── tts_output/
    └── {segment_id}.wav                  # per-segment TTS audio

logs/
├── pipeline_{episode_id}_{timestamp}.log
├── current_stage.txt
├── mixing_report_{episode_id}.json
└── output_validation_{episode_id}.json

scripts/
├── preprocessing/
│   ├── separate_audio.py                 # NEW: Demucs wrapper + SNR check
│   ├── srt_parser.py                     # NEW: SRT parsing + offset correction
│   └── dubbing_adapter.py               # NEW: localization adapter
├── inference/
│   ├── character_voice_mapper.py         # NEW: voice assignment + persistence
│   ├── subtitle_processor.py             # NEW: classifier + splitter + subtitle writer
│   ├── ocr_engine.py                     # NEW: OCR fallback (easyocr + frame extraction)
│   ├── mux_output.py                     # NEW: FFmpeg muxer
│   ├── validate_output.py               # NEW: FFprobe validator
│   └── pipeline_cache.py                # NEW: caching layer (translations/embeddings/tts)
└── training/
    ├── finetune_nemo.py                  # NEW: NeMo fine-tuning
    └── dataset_stats.py                  # NEW: dataset statistics

outputs/
└── {episode_id}.hi.srt                   # NEW: Hindi subtitle file (SRT or ASS)

data/
├── processed/
│   ├── vocals.wav
│   ├── background.wav
│   ├── separation_quality.json
│   ├── final_hindi_dubbed.wav
│   ├── ocr_signs.json                    # NEW: OCR-detected sign regions (if --ocr-signs)
│   └── frames/                           # NEW: extracted frames for OCR fallback
│       └── frame_%04d.jpg
```

---

## 15. Technology Stack and Library Choices

| Component | Library / Tool | Justification |
|-----------|---------------|---------------|
| Audio source separation | `demucs` (htdemucs model) | State-of-the-art music/vocal separation; GPU-accelerated; Python API |
| LUFS normalization | `pyloudnorm` | ITU-R BS.1770 compliant; pure Python; no FFmpeg dependency for loudness |
| Time-stretching | `librosa.effects.time_stretch` | Phase-vocoder based; preserves pitch; already in requirements.txt |
| Speaker diarization | `pyannote.audio` | Best-in-class diarization; speaker embeddings built-in |
| Voice embeddings (SRT path / fast-mode) | `resemblyzer` | Lightweight d-vector embeddings; fast CPU inference (~20-30s vs 2-5min) |
| TTS primary | `nemo_toolkit` (FastPitch+HiFiGAN) | Highest quality; CUDA-accelerated; Hindi support |
| TTS secondary | `TTS` (Coqui VITS) | Open-source; fine-tunable; good Hindi quality |
| TTS fallback | `gTTS` | Zero-dependency CPU fallback; always available |
| High-emotion TTS | `bark` | Expressive prosody; emotion presets |
| Emotion detection | `transformers` (distilroberta) | Pre-trained; fast inference; good multilingual support |
| Translation (primary) | `Helsinki-NLP/opus-mt-ja-hi` | Offline-capable; locked per episode for consistency |
| Translation (fallback) | DeepL API / Google Translate | Used only if Helsinki-NLP fails |
| Localization adapter | OpenAI GPT-4o API → Ollama → rule-based | Priority chain; GPT-4o if key set, else Ollama, else rule-based |
| Muxing | `ffmpeg` (subprocess) | Industry standard; stream copy avoids re-encoding |
| Output validation | `ffprobe` (subprocess) | Reliable media inspection; JSON output |
| API | `fastapi` + `uvicorn` | Already in use; async-friendly |
| UI | `streamlit` | Already in use; rapid iteration |
| Audio I/O | `soundfile`, `numpy` | Fast WAV read/write; float32 arrays |
| OCR (sign fallback) | `easyocr` | Multi-language OCR; Japanese model; bounding box output |
| ASS subtitle output | Manual string generation | No external dependency; full control over style blocks |
| Cache hashing | `hashlib` (SHA256) | Built-in; deterministic; collision-resistant |

---

## 16. Caching System

The pipeline maintains a `data/cache/` directory to avoid redundant computation across runs.

### Cache Directory Structure

```
data/cache/
├── translations/    # {sha256_key}.json  — cached translation results
├── embeddings/      # {sha256_key}.npy   — cached Resemblyzer embeddings
└── tts/             # {sha256_key}.wav   — cached TTS synthesis output
```

### Cache Key Computation

| Stage | Cache Key Input | File Type |
|-------|----------------|-----------|
| Translation | SHA256(source_text + src_lang + tgt_lang + backend_name) | `.json` |
| Embeddings | SHA256(audio_file_path + str(file_mtime)) | `.npy` |
| TTS | SHA256(dubbed_text + voice_profile_id + emotion + str(emotion_intensity) + tts_backend) | `.wav` |

### Cache Behavior

```python
import hashlib, json, os
import numpy as np

def cache_key(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode()).hexdigest()

def get_translation_cache(src_text, src_lang, tgt_lang, backend):
    key = cache_key(src_text, src_lang, tgt_lang, backend)
    path = f"data/cache/translations/{key}.json"
    if os.path.exists(path):
        log(f"[CACHE HIT] translation {key[:8]}")
        return json.load(open(path))
    return None

def set_translation_cache(src_text, src_lang, tgt_lang, backend, result):
    key = cache_key(src_text, src_lang, tgt_lang, backend)
    path = f"data/cache/translations/{key}.json"
    json.dump(result, open(path, "w"), ensure_ascii=False)
```

- On cache hit: log `"[CACHE HIT] {stage} {key[:8]}"` and return cached result.
- Cache is NOT invalidated automatically — user must run `--clear-cache` to clear.
- `--clear-cache` flag deletes all files in `data/cache/` subdirectories and logs the count deleted.

### CLI Flag

```
run_pipeline.py --clear-cache
```
Deletes all files in `data/cache/translations/`, `data/cache/embeddings/`, `data/cache/tts/` and logs: `"Cleared N cache files from data/cache/"`.

---

## 17. Failure Recovery — Resume from Stage

The pipeline supports resuming from any stage after a failure or interruption.

### Checkpoint File

Written to `data/processed/{episode_id}_checkpoint.json` after each stage completes successfully.

**Schema:**
```json
{
  "episode_id": "ep01",
  "completed_stages": [1, 2, 3],
  "last_stage": 3,
  "timestamp": "2025-01-01T12:34:56Z",
  "input_file": "episodes/ep01.mp4"
}
```

### Resume Logic

```python
def determine_start_stage(episode_id, resume_from=None, force_restart=False):
    checkpoint_path = f"data/processed/{episode_id}_checkpoint.json"

    if force_restart:
        log("--force-restart: ignoring checkpoint, running from stage 1")
        return 1

    if resume_from is not None:
        log(f"--resume-from {resume_from}: skipping stages 1–{resume_from - 1}")
        return resume_from

    if os.path.exists(checkpoint_path):
        checkpoint = json.load(open(checkpoint_path))
        next_stage = checkpoint["last_stage"] + 1
        log(f"Auto-resume: checkpoint found, resuming from stage {next_stage}")
        return next_stage

    return 1  # normal start
```

### Stage Input Validation

Before executing each stage, the pipeline validates that required input files exist:

```python
STAGE_INPUTS = {
    2: ["data/raw_audio/extracted.wav"],
    3: ["data/processed/vocals.wav"],
    4: ["data/processed/diarization.json"],
    5: ["data/processed/transcript_hi.json"],
    6: ["data/processed/transcript_hi.json"],
    7: ["data/voice_maps/{episode_id}_voice_map.json"],
    8: ["data/tts_output/"],
    9: ["data/processed/final_hindi_dubbed.wav"],
    10: ["outputs/{episode_id}_dubbed.mp4"],
}

def validate_stage_inputs(stage, episode_id):
    missing = []
    for path_template in STAGE_INPUTS.get(stage, []):
        path = path_template.format(episode_id=episode_id)
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise PipelineResumeError(
            f"Stage {stage} cannot resume: missing input files: {missing}"
        )
```

### CLI Flags

| Flag | Behavior |
|------|----------|
| `--resume-from N` | Skip stages 1 through N-1, start from stage N |
| `--force-restart` | Ignore existing checkpoint, run from stage 1 |
| `--fast-mode` | Use Resemblyzer for diarization (skip pyannote) |
| `--clear-cache` | Delete all cache files and exit |
| `--subtitle-offset N` | Apply N ms offset to all SRT timecodes |
