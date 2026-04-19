# Phase 2 — Quality Improvements: Tasks

- [ ] 1. Speaker voice embedding extraction and application
  - [ ] 1.1 Run `scripts/training/extract_voice_embeddings.py` to generate per-speaker embeddings
    - Run: `python scripts/training/extract_voice_embeddings.py --audio data/processed/audio.wav --diarization data/processed/diarization.json --output-dir data/voice_references/embeddings/`
    - Verify `data/voice_references/embeddings/SPEAKER_00.npy` files exist
    - Check `speaker_manifest.json` — each speaker needs `total_audio_seconds > 3.0`
    - **Done when:** Every detected speaker has a `.npy` embedding file
    - _Requirements: 1_
  - [ ] 1.2 Apply speaker embeddings in TTS (`scripts/inference/tts_hindi.py`)
    - Add `diarization_json: str = None` parameter to `synthesize_hindi()`
    - Load diarization data at function start
    - For each segment, find overlapping speaker via timestamp comparison
    - Load matching `data/voice_references/embeddings/{speaker_id}.npy`
    - Pass as `speaker_embedding` or `speaker_wav` to Coqui TTS
    - Log `speaker_id` per segment into output manifest
    - **Done when:** Different characters have noticeably different voice qualities in output
    - _Requirements: 2_

---

- [ ] 2. Emotion detection and expressive TTS
  - [ ] 2.1 Create `scripts/preprocessing/detect_emotion.py` for text-based emotion classification
    - Use `j-hartmann/emotion-english-distilroberta-base` from Hugging Face `transformers`
    - Add `emotion` (string) and `emotion_intensity` (0.0–1.0 float) to each segment
    - Map emotions to: neutral, happy, sad, angry, surprised, fearful
    - Accept `--input` and `--output` argparse arguments
    - **Done when:** Every segment in output JSON has `emotion` and `emotion_intensity` fields
    - _Requirements: 3_
  - [ ] 2.2 Add Bark TTS for high-emotion segments in `scripts/inference/tts_hindi.py`
    - Install: `pip install git+https://github.com/suno-ai/bark.git`
    - Add branch: if `emotion_intensity > 0.7`, use Bark with `BARK_VOICE_MAP` per emotion
    - Otherwise fall back to Coqui TTS
    - Log `tts_engine` as `"bark"` or `"coqui"` in the manifest
    - **Note:** Bark is ~5× slower than Coqui — only for high-emotion branch
    - **Done when:** High-emotion segments sound more expressive; `tts_engine` field logged
    - _Requirements: 3, 4_

---

- [ ] 3. Prosody transfer and alignment improvements
  - [ ] 3.1 Create `scripts/inference/prosody_transfer.py` for pitch contour transfer
    - Use `pyworld` to extract F0 from both Japanese original and Hindi TTS
    - Normalize both F0 curves (remove speaker-specific mean/std)
    - Resample original F0 shape to match TTS length
    - Scale to TTS speaker's mean/std and synthesize
    - Wire into `align_and_mix.py` as optional `--prosody-transfer` flag
    - **Done when:** Hindi audio follows emotional pitch arc of original Japanese; PESQ score doesn't drop
    - _Requirements: 5_
  - [ ] 3.2 Upgrade to DTW-based alignment in `scripts/inference/align_and_mix.py`
    - Replace uniform time-stretch with DTW using `librosa.sequence.dtw()` on MFCC features
    - Map warping path frame indices back to sample indices and interpolate
    - Keep old uniform stretch as fallback if DTW stretch ratio > 1.5 or < 0.6
    - **Done when:** Aligned audio sounds more natural; fewer "chipmunk" or "slow-motion" artifacts
    - _Requirements: 6_

---

- [ ] 4. Subtitle generation
  - [ ] 4.1 Run `scripts/inference/generate_subtitles.py` and validate SRT output
    - Run: `python scripts/inference/generate_subtitles.py --input data/processed/transcript_hi.json --output outputs/subtitles_hi.srt`
    - Open in VLC (Subtitles → Add Subtitle File) to verify Hindi text at correct timestamps
    - Verify format: index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text, blank line
    - **Done when:** Valid `.srt` file, subtitles visible in VLC at correct times
    - _Requirements: 7_
