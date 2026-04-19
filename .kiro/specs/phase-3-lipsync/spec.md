# Spec: Phase 3 — Video Output (Lip Sync)

## Overview
Add lip-synchronized video output to the pipeline. The dubbed Hindi audio is used to drive mouth animation on the anime character video. Phase 1 and Phase 2 must be complete before starting this phase.

**Prerequisite:** `outputs/final_hi_dub.wav` exists.

---

## Requirements

1. Wav2Lip is installed and its pretrained weights are available at `models/lip_sync/wav2lip.pth`
2. A `scripts/inference/lip_sync.py` script wraps Wav2Lip into the pipeline interface
3. The master `run_pipeline.py` calls lip sync as Stage 7 when `--video-output` flag is set
4. For anime/stylized faces, SadTalker is evaluated as an alternative to Wav2Lip

---

## Tasks

### Task 1: Install Wav2Lip
Clone and install Wav2Lip.

```bash
git clone https://github.com/Rudrabha/Wav2Lip
cd Wav2Lip
pip install -r requirements.txt
```

Download pretrained weights from the Wav2Lip README instructions and save them to `models/lip_sync/wav2lip.pth`.

Test with a sample realistic face video:
```bash
cd Wav2Lip
python inference.py \
  --checkpoint_path ../models/lip_sync/wav2lip.pth \
  --face sample_face.mp4 \
  --audio ../outputs/final_hi_dub.wav \
  --outfile ../outputs/test_lipsync.mp4
```

**Done when:** `outputs/test_lipsync.mp4` plays with lip movements that roughly match the dubbed audio.

---

### Task 2: Create Lip Sync Integration Script
Create `scripts/inference/lip_sync.py` that wraps Wav2Lip into the project's script interface.

```python
import subprocess, sys
from pathlib import Path

def run_lip_sync(
    face_video: str,
    audio_path: str,
    output_path: str,
    checkpoint: str = "models/lip_sync/wav2lip.pth",
    wav2lip_dir: str = "Wav2Lip"
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        f"{wav2lip_dir}/inference.py",
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
    ]
    result = subprocess.run(cmd, check=True)
    print(f"[LipSync] Output saved: {output_path}")
    return output_path
```

Add `--input` (face video), `--audio`, `--output` argparse arguments so it can be called standalone.

**Done when:** `python scripts/inference/lip_sync.py --input face.mp4 --audio outputs/final_hi_dub.wav --output outputs/final_hi_video.mp4` runs without error and produces a video file.

---

### Task 3: Wire Lip Sync into run_pipeline.py
Edit `scripts/inference/run_pipeline.py` to add Stage 7.

Add a `--video-output` flag and a `--face-video` flag. At the end of the pipeline, if `--video-output` is set:

```python
if args.video_output:
    if not args.face_video:
        print("[Pipeline] --face-video required for --video-output. Skipping lip sync.")
    else:
        run_stage(
            "Stage 7: Lip Sync",
            "scripts/inference/lip_sync.py",
            ["--input", args.face_video,
             "--audio", f"outputs/final_{args.lang}_dub.wav",
             "--output", f"outputs/final_{args.lang}_video.mp4"]
        )
```

**Done when:** `python scripts/inference/run_pipeline.py --input ep.mp4 --lang hi --video-output --face-video face.mp4` runs all 7 stages and produces `outputs/final_hi_video.mp4`.

---

### Task 4: Evaluate SadTalker for Anime Faces
Wav2Lip is optimized for realistic faces. Anime faces need a different model.

Evaluate SadTalker (https://github.com/OpenTalker/SadTalker):
```bash
git clone https://github.com/OpenTalker/SadTalker
cd SadTalker
# Follow their README for model download + install
```

Test it on an anime character still image + the dubbed audio:
```bash
python inference.py \
  --driven_audio ../outputs/final_hi_dub.wav \
  --source_image anime_character.png \
  --result_dir ../outputs/sadtalker/
```

Compare output quality between Wav2Lip (requires video) and SadTalker (works with a single image).

Write a short `docs/lipsync_comparison.md` noting which model works better for anime style, with sample output descriptions.

**Done when:** Both models tested, `docs/lipsync_comparison.md` written with recommendation.

---

## Acceptance Criteria

- [ ] `models/lip_sync/wav2lip.pth` — weights file exists
- [ ] `outputs/test_lipsync.mp4` — plays with synchronized lip movements
- [ ] `scripts/inference/lip_sync.py` — runs standalone with `--input`, `--audio`, `--output` args
- [ ] `run_pipeline.py` — `--video-output` flag triggers Stage 7 without crashing
- [ ] `outputs/final_hi_video.mp4` — produced by full pipeline run with `--video-output`
- [ ] SadTalker tested on at least one anime character image
- [ ] `docs/lipsync_comparison.md` — exists with Wav2Lip vs SadTalker recommendation
