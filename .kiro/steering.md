# Anime Dub AI — Kiro Steering Document

## Project Overview
An end-to-end AI pipeline that transforms Japanese-dubbed anime audio into natural Hindi voices (Phase 1), then expands to Haryanvi, Punjabi, Tamil, and other Indian languages.

## Tech Stack
- **Runtime:** Python 3.10+
- **ML Framework:** PyTorch 2.1+
- **ASR:** faster-whisper (Whisper large-v3)
- **Translation:** deep-translator (Google Translate)
- **TTS:** Coqui TTS (VITS Hindi model), Bark for expressive lines
- **Voice Cloning:** Resemblyzer + pyannote.audio
- **Audio Processing:** librosa, pydub, pyworld, noisereduce
- **API:** FastAPI + uvicorn
- **UI:** Streamlit
- **Evaluation:** PESQ, STOI

## Repository Layout
```
anime-dub-ai/
├── .kiro/specs/              ← Kiro task specs (this folder)
├── configs/                  ← YAML configs (source of truth)
├── data/                     ← Audio data (gitignored)
├── scripts/
│   ├── preprocessing/        ← Stages 1–3: extract, diarize, ASR, translate
│   ├── inference/            ← Stages 4–6: TTS, align, mix, subtitles
│   ├── training/             ← Embedding extraction, fine-tuning
│   ├── evaluation/           ← Quality metrics
│   └── dialects/             ← Dialect post-processors
├── api/                      ← FastAPI REST server
├── ui/                       ← Streamlit demo app
├── models/                   ← Model weights (gitignored)
└── outputs/                  ← Final dubbed audio (gitignored)
```

## Pipeline Stages (Sequential)
| # | Script | Input → Output |
|---|--------|----------------|
| 1 | `extract_audio.py` | `.mp4` → `data/processed/audio.wav` |
| 1b | `diarize_speakers.py` | `audio.wav` → `diarization.json` |
| 2 | `asr_transcribe.py` | `audio.wav` → `transcript_ja.json` |
| 3 | `translate.py` | `transcript_ja.json` → `transcript_hi.json` |
| 4 | `tts_hindi.py` | `transcript_hi.json` → `tts_output/` wavs |
| 5 | `align_and_mix.py` | `segments.json` → `outputs/final_hi_dub.wav` |

**Full run:** `python scripts/inference/run_pipeline.py --input episode.mp4 --lang hi`

## Coding Conventions
- All scripts use `argparse` with consistent `--input`, `--output` flags
- All scripts print stage name in `[BracketFormat]` at key steps
- All configs read from `configs/pipeline_config.yaml` or `configs/languages.yaml`
- GPU auto-detection: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- No hardcoded paths — use `pathlib.Path` throughout
- `data/`, `models/`, `outputs/` are gitignored; never commit binary files

## Key Config Files
- `configs/pipeline_config.yaml` — source of truth for all stage parameters
- `configs/languages.yaml` — add a new language here, no code changes needed
- `configs/character_voices.yaml` — per-character voice profiles for cloning

## Phase Status
- ✅ **Phase 0:** All base files implemented
- 🔧 **Phase 1:** Core pipeline — verify and wire up end-to-end
- 🔧 **Phase 2:** Quality — voice cloning, emotion, prosody, DTW
- 📋 **Phase 3:** Lip sync — Wav2Lip / SadTalker integration
- 📋 **Phase 4:** Multi-language — Haryanvi, Punjabi, Tamil
- 📋 **Phase 5:** UI/API — FastAPI + Streamlit
- 📋 **Phase 6:** Fine-tuning — custom VITS models
- 📋 **Phase 7:** Tests + CI
