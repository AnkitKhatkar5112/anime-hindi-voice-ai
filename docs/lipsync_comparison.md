# Lip Sync Model Comparison: Wav2Lip vs SadTalker

**Project:** Anime Dub AI — Phase 3 Lip Sync Evaluation  
**Date:** 2025  
**Requirement:** Req 4 — Evaluate SadTalker as an alternative to Wav2Lip for anime/stylized faces

---

## Overview

Two models were evaluated for driving lip-sync on anime characters using Hindi dubbed audio:

| | Wav2Lip | SadTalker |
|---|---|---|
| **Paper** | Wav2Lip (ACMMM 2020) | SadTalker (CVPR 2023) |
| **Repo** | [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | [OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker) |
| **Input** | Video + Audio | Single image + Audio |
| **Output** | Video with synced lips | Full talking-head video |
| **Face type** | Realistic (trained on LRS2/LRS3) | Realistic + stylized |

---

## Wav2Lip

### How it works
Wav2Lip uses a lip-sync discriminator trained on real talking-head videos (LRS2, LRS3 datasets). It replaces the lower half of the face in each video frame with a generated lip region that matches the audio mel-spectrogram.

### Input requirements
- **Source video** — a video of the character's face is mandatory. A still image alone is not sufficient.
- Audio file (.wav)
- Pretrained checkpoint (`wav2lip.pth` or `wav2lip_gan.pth`)

### Strengths
- Very accurate phoneme-to-lip alignment on realistic faces
- Fast inference (real-time capable on GPU)
- Well-documented, widely used

### Weaknesses for anime
- **Trained exclusively on realistic human faces** — the face detector (S3FD) often fails to detect anime-style faces, which lack realistic skin texture, proportions, and shading
- Even when detection succeeds, the generated lip region looks like a photorealistic mouth pasted onto a cartoon face — a jarring visual mismatch
- Requires a source video, which may not exist for anime characters (often only key-art or still frames are available)
- The GAN variant (`wav2lip_gan.pth`) produces sharper results but still suffers from the domain mismatch

### Sample output description (realistic face)
On a realistic talking-head video, Wav2Lip produces clean, well-synced lip movements with minimal artifacts. The transition between original and generated regions is smooth. Sync quality is excellent.

### Sample output description (anime face)
On an anime character still (converted to a looping video), the S3FD face detector frequently returns no detections or a bounding box misaligned with the drawn face. When inference completes, the output shows a blurry, photorealistic mouth region overlaid on the cartoon face — visually inconsistent and unsuitable for production use.

---

## SadTalker

### How it works
SadTalker generates a 3D motion coefficient sequence from audio (using a learned audio-to-expression mapping), then renders a talking-head video using a 3D Morphable Model (3DMM) and a face renderer. It can animate from a single still image.

### Input requirements
- **Single still image** — no source video needed. Works with PNG/JPG artwork.
- Audio file (.wav)
- Pretrained weights (downloaded via `bash scripts/download_models.sh`)

### Strengths
- **Works from a single image** — ideal for anime where only key-art exists
- The 3DMM-based approach is more style-agnostic than Wav2Lip's discriminator
- `--still` mode reduces exaggerated head motion, producing calmer animations suitable for anime
- `--preprocess full` handles images that aren't tightly cropped to a face
- Optional GFPGAN enhancer sharpens output frames
- Generates natural head pose and eye blink alongside lip movement

### Weaknesses for anime
- The face landmark detector (used internally) can still struggle with highly stylized anime faces (very large eyes, simplified nose/mouth)
- Output resolution is limited to 256×256 or 512×512 — upscaling may be needed for HD output
- Inference is slower than Wav2Lip (~30–60s per 10s clip on a mid-range GPU)
- The rendered face is warped from the source image, which can introduce slight texture distortion on detailed anime artwork

### Sample output description (anime face)
On an anime character still image with `--still --preprocess full --size 256`, SadTalker produces a video where the character's mouth opens and closes in rough sync with the Hindi audio. Head motion is minimal. The overall style remains consistent with the source artwork — no photorealistic region is pasted in. Lip shapes are approximate (not phoneme-perfect) but visually acceptable for anime dubbing where stylized mouth movement is the norm.

---

## Practical Comparison

| Criterion | Wav2Lip | SadTalker |
|---|---|---|
| Requires source video | ✅ Yes (mandatory) | ❌ No (single image) |
| Works with anime art | ❌ Poor (domain mismatch) | ✅ Acceptable |
| Lip sync accuracy | ⭐⭐⭐⭐⭐ (realistic faces) | ⭐⭐⭐ (all faces) |
| Style consistency | ❌ Pastes realistic mouth | ✅ Preserves art style |
| Model size | ~400 MB | ~1.5 GB (multiple weights) |
| Inference speed | Fast (~real-time) | Moderate (30–60s/10s clip) |
| GPU requirement | 4 GB VRAM | 6–8 GB VRAM recommended |
| Setup complexity | Low | Medium (multiple weight files) |
| Single-image support | ❌ | ✅ |

---

## Recommendation

**Use SadTalker for anime characters.**

Wav2Lip is the better model for realistic talking-head video (e.g., dubbing a live-action interview), but it is not suitable for anime-style artwork due to training domain mismatch and the hard requirement for a source video.

SadTalker's single-image input and style-preserving renderer make it the practical choice for this project. The lip sync accuracy is lower than Wav2Lip on realistic faces, but for anime dubbing — where audiences are accustomed to stylized, approximate mouth movement — the output is visually coherent and production-usable.

### Recommended SadTalker settings for anime

```bash
python scripts/inference/sad_talker.py \
    --source-image anime_character.png \
    --audio outputs/final_hi_dub.wav \
    --result-dir outputs/sadtalker/ \
    --preprocess full \
    --size 256 \
    # --enhancer gfpgan   # add for sharper output (requires GFPGAN weights)
```

Or using SadTalker directly:

```bash
cd SadTalker
python inference.py \
    --driven_audio ../outputs/final_hi_dub.wav \
    --source_image ../anime_character.png \
    --result_dir ../outputs/sadtalker/ \
    --still \
    --preprocess full \
    --size 256
```

---

## Setup Instructions

### Wav2Lip

```bash
git clone https://github.com/Rudrabha/Wav2Lip
cd Wav2Lip
pip install -r requirements.txt
# Download wav2lip.pth from the README Google Drive link
# Place at: models/lip_sync/wav2lip.pth

python scripts/inference/lip_sync.py \
    --input face_video.mp4 \
    --audio outputs/final_hi_dub.wav \
    --output outputs/final_hi_video.mp4
```

### SadTalker

```bash
git clone https://github.com/OpenTalker/SadTalker
cd SadTalker
pip install -r requirements.txt
bash scripts/download_models.sh   # downloads ~1.5 GB of weights

python scripts/inference/sad_talker.py \
    --source-image anime_character.png \
    --audio outputs/final_hi_dub.wav \
    --result-dir outputs/sadtalker/
```

---

## Future Improvements

- **SadTalker + anime fine-tuning:** Fine-tuning the audio-to-expression mapping on anime face datasets could improve phoneme accuracy for stylized faces.
- **DiffTalk / SyncTalk:** Newer diffusion-based talking-head models may offer better quality for stylized faces — worth evaluating in a future phase.
- **Hybrid approach:** Use Wav2Lip for any realistic face footage (e.g., live-action inserts) and SadTalker for anime character stills within the same episode.
