# Phase 3 — Video Output (Lip Sync): Tasks

- [ ] 1. Wav2Lip installation and baseline test
  - [ ] 1.1 Clone and install Wav2Lip with pretrained weights
    - Run: `git clone https://github.com/Rudrabha/Wav2Lip && cd Wav2Lip && pip install -r requirements.txt`
    - Download pretrained weights from Wav2Lip README, save to `models/lip_sync/wav2lip.pth`
    - _Requirements: 1_
  - [ ] 1.2 Test Wav2Lip on a sample realistic face video
    - Run: `python Wav2Lip/inference.py --checkpoint_path models/lip_sync/wav2lip.pth --face sample_face.mp4 --audio outputs/final_hi_dub.wav --outfile outputs/test_lipsync.mp4`
    - **Done when:** `outputs/test_lipsync.mp4` plays with lip movements roughly matching dubbed audio
    - _Requirements: 1_

---

- [ ] 2. Lip sync pipeline integration
  - [ ] 2.1 Create `scripts/inference/lip_sync.py` wrapper script
    - Implement `run_lip_sync(face_video, audio_path, output_path, checkpoint, wav2lip_dir)` via subprocess
    - Add `--input` (face video), `--audio`, `--output` argparse arguments for standalone use
    - Auto-create output parent directory with `Path.mkdir(parents=True, exist_ok=True)`
    - **Done when:** `python scripts/inference/lip_sync.py --input face.mp4 --audio outputs/final_hi_dub.wav --output outputs/final_hi_video.mp4` runs without error
    - _Requirements: 2_
  - [ ] 2.2 Wire lip sync into `scripts/inference/run_pipeline.py` as Stage 7
    - Add `--video-output` and `--face-video` CLI flags
    - If `--video-output` is set and `--face-video` provided, run Stage 7: Lip Sync
    - Output to `outputs/final_{lang}_video.mp4`
    - Print warning if `--video-output` set without `--face-video`
    - **Done when:** Full pipeline with `--video-output --face-video face.mp4` produces `outputs/final_hi_video.mp4`
    - _Requirements: 3_

---

- [ ] 3. SadTalker evaluation for anime faces
  - [ ] 3.1 Clone, install, and test SadTalker on anime character
    - Clone: `git clone https://github.com/OpenTalker/SadTalker` and follow README for model download
    - Test on anime character still image + dubbed audio
    - Run: `python inference.py --driven_audio outputs/final_hi_dub.wav --source_image anime_character.png --result_dir outputs/sadtalker/`
    - _Requirements: 4_
  - [ ] 3.2 Write comparison document `docs/lipsync_comparison.md`
    - Compare Wav2Lip (requires video) vs SadTalker (works with single image)
    - Note which model works better for anime style with sample output descriptions
    - **Done when:** `docs/lipsync_comparison.md` written with recommendation
    - _Requirements: 4_
