# Phase 1 — Core Pipeline (Hindi): Tasks

- [-] 1. Environment setup and dependency installation
  - [x] 1.1 Install all packages from `requirements.txt`
    - Run `pip install -r requirements.txt`
    - If `torch` fails due to CUDA mismatch, install the correct build from https://pytorch.org/get-started/locally/
    - _Requirements: 1_
  - [x] 1.2 Configure `.env` with HuggingFace token
    - Copy `.env.example` to `.env`
    - Generate HF token at https://huggingface.co/settings/tokens
    - Accept pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
    - Add `HF_TOKEN=hf_your_token_here` to `.env`
    - _Requirements: 2_

---

- [x] 2. Audio extraction and preprocessing
  - [x] 2.1 Test `scripts/preprocessing/extract_audio.py` on a sample clip
    - Run: `python scripts/preprocessing/extract_audio.py --input sample.mp4 --output data/processed/audio.wav`
    - Verify: file exists, plays correctly, is 22050 Hz mono
    - Use `librosa.get_duration()` to confirm duration matches source
    - **Done when:** Clean WAV produced with no ffmpeg errors and correct properties
    - _Requirements: 3_

---

- [x] 3. Speaker diarization testing
  - [x] 3.1 Run `scripts/preprocessing/diarize_speakers.py` and verify speaker segments
    - Run: `python scripts/preprocessing/diarize_speakers.py --audio data/processed/audio.wav --output data/processed/diarization.json`
    - Verify JSON contains entries like `{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00", "duration": 2.7}`
    - **Done when:** JSON has 2+ distinct `SPEAKER_*` labels with non-overlapping time ranges
    - _Requirements: 4_

---

- [x] 4. Japanese ASR transcription
  - [x] 4.1 Run `scripts/preprocessing/asr_transcribe.py` and verify Japanese transcript
    - Run: `python scripts/preprocessing/asr_transcribe.py --audio data/processed/audio.wav --model medium --output data/processed/transcript_ja.json`
    - Use `medium` model for speed testing, switch to `large-v3` for production
    - **Done when:** `transcript_ja.json` contains Japanese text with `start`, `end`, and `words` per segment
    - _Requirements: 5_
  - [x] 4.2 Fix GPU/CPU auto-detection in `scripts/preprocessing/asr_transcribe.py`
    - Replace hardcoded `device="cuda"` with `torch.cuda.is_available()` check
    - Use `compute_type="float16"` for CUDA, `"int8"` for CPU
    - Add a `--device` CLI argument for manual override
    - Print which device is being used at startup
    - **Done when:** Script runs on both GPU and CPU-only machines without manual config edits
    - _Requirements: 5_

---

- [x] 5. Translation pipeline (Japanese → Hindi)
  - [x] 5.1 Run `scripts/preprocessing/translate.py` to translate Japanese transcript
    - Run: `python scripts/preprocessing/translate.py --input data/processed/transcript_ja.json --output data/processed/transcript_hi.json`
    - **Done when:** Every segment has `text_original` (Japanese) and `text_translated` (Hindi) fields
    - Manually spot-check 5–10 lines
    - _Requirements: 6_
  - [x] 5.2 Add Hindi text cleaning post-processing to `scripts/preprocessing/translate.py`
    - Add `clean_hindi_text()` function with regex whitespace cleanup + anime term map
    - Call after translation, save result as `text_cleaned` alongside `text_translated`
    - **Done when:** Output JSON has both `text_translated` and `text_cleaned` fields per segment
    - _Requirements: 6, 10_

---

- [x] 6. Hindi TTS synthesis
  - [x] 6.1 Run `scripts/inference/tts_hindi.py` for Hindi speech synthesis
    - Run: `python scripts/inference/tts_hindi.py --input data/processed/transcript_hi.json --output-dir data/tts_output/`
    - First run downloads Coqui Hindi VITS model (~150 MB)
    - If CPU-only: change `.to("cuda")` to `.to("cpu")` in `synthesize_hindi()`
    - **Done when:** Files `data/tts_output/seg_0000.wav` through `seg_NNNN.wav` exist and are intelligible Hindi
    - _Requirements: 7_
  - [x] 6.2 Add TTS stretch ratio logging to `scripts/inference/tts_hindi.py`
    - Compute `tts_duration`, `original_duration`, and `stretch_ratio` per segment after synthesis
    - Save to `data/tts_output/segments.json`
    - **Done when:** `segments.json` has `stretch_ratio` per segment, most values between 0.7 and 1.5
    - _Requirements: 7, 11_

---

- [x] 7. Final alignment, mixing, and end-to-end pipeline
  - [x] 7.1 Test `scripts/inference/align_and_mix.py` for final audio assembly
    - Get source duration with `librosa.get_duration(path="data/processed/audio.wav")`
    - Run: `python scripts/inference/align_and_mix.py --segments data/tts_output/segments.json --output outputs/final_hi_dub.wav --duration <total_seconds>`
    - **Done when:** `outputs/final_hi_dub.wav` plays without silence gaps, clipping, or crashes; duration within 5% of source
    - _Requirements: 8_
  - [x] 7.2 Run full end-to-end pipeline with `run_pipeline.py`
    - Run: `python scripts/inference/run_pipeline.py --input sample.mp4 --lang hi`
    - Test `--start-stage 3` resume from Stage 3
    - Test `--skip-diarize` for faster iteration
    - Test `--model-size medium` for speed testing
    - **Done when:** All 6 stages complete, `outputs/final_hi_dub.wav` exists and plays back Hindi dialogue with correct timing
    - _Requirements: 9_
