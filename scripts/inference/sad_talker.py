"""
SadTalker Wrapper — runs SadTalker inference via subprocess.

SadTalker generates a talking-head video from a single still image + audio,
making it ideal for anime characters where no source video is available.

Usage:
    python scripts/inference/sad_talker.py \
        --source-image anime_character.png \
        --audio outputs/final_hi_dub.wav \
        --result-dir outputs/sadtalker/

Setup (run once before using this script):
    git clone https://github.com/OpenTalker/SadTalker
    cd SadTalker
    pip install -r requirements.txt
    # Download pretrained weights per SadTalker README:
    #   bash scripts/download_models.sh
    # Or manually place weights in SadTalker/checkpoints/ and SadTalker/gfpgan/weights/
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_sad_talker(
    source_image: str,
    audio_path: str,
    result_dir: str,
    sadtalker_dir: str = "SadTalker",
    still: bool = True,
    preprocess: str = "full",
    enhancer: str | None = None,
    size: int = 256,
) -> None:
    """
    Run SadTalker inference to animate source_image with audio_path.

    Args:
        source_image: Path to the source still image (PNG/JPG). Works with
                      anime-style artwork — no video required.
        audio_path:   Path to the driving audio (.wav).
        result_dir:   Directory where SadTalker saves the output video.
        sadtalker_dir: Path to the cloned SadTalker repository.
        still:        Use still-mode (less head motion). Recommended for anime.
        preprocess:   Face crop mode — "crop", "resize", or "full".
                      Use "full" for anime illustrations that lack a tight face crop.
        enhancer:     Optional face enhancer — "gfpgan" improves output sharpness.
                      Requires GFPGAN weights (see SadTalker README).
        size:         Output resolution (256 or 512). 512 gives higher quality.
    """
    sadtalker_path = Path(sadtalker_dir)
    if not sadtalker_path.exists():
        print(
            f"[SadTalker] ❌ SadTalker directory not found: {sadtalker_dir}\n"
            "  Clone it first:\n"
            "    git clone https://github.com/OpenTalker/SadTalker\n"
            "    cd SadTalker && bash scripts/download_models.sh"
        )
        sys.exit(1)

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    inference_script = str(sadtalker_path / "inference.py")
    cmd = [
        sys.executable, inference_script,
        "--driven_audio", audio_path,
        "--source_image", source_image,
        "--result_dir", result_dir,
        "--preprocess", preprocess,
        "--size", str(size),
    ]

    if still:
        cmd.append("--still")

    if enhancer:
        cmd.extend(["--enhancer", enhancer])

    print(f"[SadTalker] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(sadtalker_path))
    if result.returncode != 0:
        print(f"[SadTalker] ❌ Inference failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"[SadTalker] ✅ Output saved to: {result_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SadTalker wrapper — animate a still image with audio (anime-friendly)"
    )
    parser.add_argument(
        "--source-image", required=True,
        help="Source still image (.png/.jpg) — anime artwork works well"
    )
    parser.add_argument(
        "--audio", required=True,
        help="Driving audio file (.wav)"
    )
    parser.add_argument(
        "--result-dir", required=True,
        help="Output directory for generated video"
    )
    parser.add_argument(
        "--sadtalker-dir", default="SadTalker",
        help="Path to cloned SadTalker repository (default: SadTalker/)"
    )
    parser.add_argument(
        "--no-still", action="store_true",
        help="Disable still-mode (allows more head motion)"
    )
    parser.add_argument(
        "--preprocess", default="full",
        choices=["crop", "resize", "full"],
        help="Face crop mode — 'full' recommended for anime illustrations"
    )
    parser.add_argument(
        "--enhancer", default=None,
        choices=["gfpgan"],
        help="Optional face enhancer for sharper output (requires GFPGAN weights)"
    )
    parser.add_argument(
        "--size", default=256, type=int,
        choices=[256, 512],
        help="Output resolution (256 or 512)"
    )
    args = parser.parse_args()

    run_sad_talker(
        source_image=args.source_image,
        audio_path=args.audio,
        result_dir=args.result_dir,
        sadtalker_dir=args.sadtalker_dir,
        still=not args.no_still,
        preprocess=args.preprocess,
        enhancer=args.enhancer,
        size=args.size,
    )


if __name__ == "__main__":
    main()
