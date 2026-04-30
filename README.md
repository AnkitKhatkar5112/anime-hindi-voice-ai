git pull


git add .
git commit -m "new-implementation task 3"
git push


# 🎌 Anime Dub AI — Japanese → Hindi (+ Multi-Language)

End-to-end AI pipeline that transforms Japanese-dubbed anime audio into natural, expressive Hindi voices — then expands to Haryanvi, Punjabi, Tamil, and more Indian languages.

---

## 🗂️ Project Structure

```
anime-dub-ai/
│
├── .kiro/                            ← Kiro AI specs (open in Kiro to start)
│   ├── steering.md                   ← Project context for the AI agent
│   ├── hooks/post-edit.md            ← Auto-checks after code edits
│   └── specs/
│       ├── phase-1-core-pipeline/    ← ✅ Start here — wire up the pipeline
│       ├── phase-2-quality/          ← Voice cloning, emotion, prosody
│       ├── phase-3-lipsync/          ← Wav2Lip / SadTalker video output
│       ├── phase-4-multilang/        ← Haryanvi, Punjabi, Tamil, etc.
│       ├── phase-5-ui-api/           ← FastAPI + Streamlit UI
│       ├── phase-6-finetuning/       ← Custom VITS model training
│       └── phase-7-testing/          ← pytest + quality benchmarks
│
├── configs/                          ← All pipeline parameters (source of truth)
│   ├── pipeline_config.yaml          ← Master config — edit this, not the code
│   ├── languages.yaml                ← Add new languages here (no code changes)
│   └── character_voices.yaml         ← Per-character voice profiles
│
├── scripts/
│   ├── preprocessing/                ← Stages 1–3
│   │   ├── extract_audio.py          ← Stage 1: video → clean 22050Hz WAV
│   │   ├── diarize_speakers.py       ← Stage 1b: who speaks when (pyannote)
│   │   ├── asr_transcribe.py         ← Stage 2: Japanese ASR (Whisper large-v3)
│   │   └── translate.py              ← Stage 3: Japanese text → Hindi text
│   │
│   ├── inference/                    ← Stages 4–6 + utilities
│   │   ├── run_pipeline.py           ← 🚀 MAIN ENTRY POINT — runs all stages
│   │   ├── tts_hindi.py              ← Stage 4: Hindi text → Hindi speech (Coqui)
│   │   ├── align_and_mix.py          ← Stage 5: time-align + BGM mix
│   │   ├── generate_subtitles.py     ← Generates .srt subtitle file
│   │   └── batch_process.py          ← Process full season of episodes
│   │
│   ├── training/
│   │   └── extract_voice_embeddings.py  ← Speaker fingerprints (Resemblyzer)
│   │
│   ├── evaluation/
│   │   └── evaluate_quality.py       ← PESQ + STOI quality metrics
│   │
│   └── dialects/
│       └── hindi_to_haryanvi.py      ← Rule-based Hindi → Haryanvi converter
│
├── api/
│   └── main.py                       ← FastAPI: POST /dub, GET /status, GET /download
│
├── ui/
│   └── app.py                        ← Streamlit demo UI
│
├── notebooks/
│   ├── 01_explore_audio.ipynb        ← Visualize waveform, spectrogram, pitch
│   └── 02_review_translations.ipynb  ← QA translated segments side-by-side
│
├── data/                             ← Runtime data (gitignored)
│   ├── raw_audio/                    ← Extracted source audio
│   ├── processed/                    ← Cleaned audio + transcript JSONs
│   ├── tts_output/                   ← Per-segment Hindi TTS wavs
│   ├── aligned/                      ← Time-aligned segments
│   └── voice_references/             ← Speaker embedding .npy files
│
├── models/                           ← Model weights (gitignored, download separately)
│   ├── asr/                          ← Whisper weights
│   ├── tts/                          ← Coqui VITS weights
│   ├── voice_clone/                  ← Resemblyzer weights
│   └── lip_sync/                     ← Wav2Lip weights
│
├── tests/
│   └── fixtures/                     ← 30-second test clip + expected outputs
│
├── outputs/                          ← Final dubbed audio/video (gitignored)
├── logs/                             ← Pipeline logs + benchmark results
├── docs/                             ← Additional documentation
├── requirements.txt
├── .env.example                      ← Copy to .env and add your tokens
└── .gitignore
```

---

## 🔄 Pipeline — How It Works

```
Input Video (.mp4 / .mkv)
        │
        ▼
[Stage 1]  extract_audio.py      → data/processed/audio.wav
[Stage 1b] diarize_speakers.py   → data/processed/diarization.json
[Stage 2]  asr_transcribe.py     → data/processed/transcript_ja.json
[Stage 3]  translate.py          → data/processed/transcript_hi.json
[Stage 4]  tts_hindi.py          → data/tts_output/  (per-segment wavs)
[Stage 5]  align_and_mix.py      → outputs/final_hi_dub.wav
        │
        ▼
  Final Hindi Dubbed Audio 🎌
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# → Add HF_TOKEN from https://huggingface.co/settings/tokens

# 3. Run the full pipeline
python scripts/inference/run_pipeline.py --input episode.mp4 --lang hi

# 4. Resume from a specific stage (e.g. skip re-extraction)
python scripts/inference/run_pipeline.py --input episode.mp4 --lang hi --start-stage 3

# 5. Batch process a full season
python scripts/inference/batch_process.py --input-dir episodes/ --lang hi --skip-processed

# 6. Run the Streamlit UI
streamlit run ui/app.py

# 7. Start the REST API
uvicorn api.main:app --reload --port 8000
```

---

## 🌐 Supported Languages

| Code | Language | Status |
|------|----------|--------|
| `hi` | Hindi (हिन्दी) | ✅ Active |
| `hry` | Haryanvi (हरयाणवी) | 🔧 In Progress |
| `pa` | Punjabi (ਪੰਜਾਬੀ) | 📋 Planned |
| `ta` | Tamil (தமிழ்) | 📋 Planned |
| `te` | Telugu (తెలుగు) | 📋 Planned |
| `bn` | Bengali (বাংলা) | 📋 Planned |

To add a new language: edit `configs/languages.yaml` only — no code changes needed.

---

## ⚙️ Key Config — pipeline_config.yaml

| Section | Key Setting | Default |
|---------|-------------|---------|
| `asr` | `model_size` | `large-v3` |
| `translation` | `service` | `google` |
| `tts` | `engine` | `coqui` |
| `alignment` | `stretch_limit` | `1.5` |
| `output` | `subtitle_output` | `true` |

---

## 🗺️ Using with Kiro

Open the project folder in **Kiro**. The `.kiro/` folder is automatically detected.

- Start at `.kiro/specs/phase-1-core-pipeline/spec.md`
- Each spec has **Requirements → Tasks → Acceptance Criteria**
- Work through phases in order: Phase 1 → 2 → 3 → ...
- `steering.md` gives Kiro full project context so it understands the codebase

---

## 📊 Evaluate Output Quality

```bash
python scripts/evaluation/evaluate_quality.py \
  --original data/processed/audio.wav \
  --dubbed outputs/final_hi_dub.wav
```

Output: PESQ score (1–4.5, higher = better), STOI score (0–1, higher = better)

---

## 💡 Tips

- Test on **30–60 second clips** before full episodes
- GPU strongly recommended for Stage 2 (ASR) and Stage 4 (TTS); CPU is ~10× slower
- Use `--skip-diarize` for faster iteration when working on translation/TTS stages
- Use `--start-stage N` to resume a failed run without reprocessing earlier stages
