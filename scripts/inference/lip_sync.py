"""
Lip Sync Wrapper — runs Wav2Lip inference via subprocess.
Usage: python scripts/inference/lip_sync.py --input face.mp4 --audio outputs/final_hi_dub.wav --output outputs/final_hi_video.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_lip_sync(
    face_video: str,
    audio_path: str,
    output_path: str,
    checkpoint: str = "models/lip_sync/wav2lip.pth",
    wav2lip_dir: str = "Wav2Lip",
):
    """Run Wav2Lip inference to sync lips in face_video to audio_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    inference_script = str(Path(wav2lip_dir) / "inference.py")
    cmd = [
        sys.executable, inference_script,
        "--checkpoint_path", checkpoint,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_path,
    ]

    print(f"  Running Wav2Lip: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Wav2Lip inference failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"  ✅ Lip sync complete → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Wav2Lip lip sync wrapper")
    parser.add_argument("--input", required=True, help="Face video file (.mp4)")
    parser.add_argument("--audio", required=True, help="Dubbed audio file (.wav)")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--checkpoint", default="models/lip_sync/wav2lip.pth",
                        help="Path to Wav2Lip checkpoint")
    parser.add_argument("--wav2lip-dir", default="Wav2Lip",
                        help="Path to Wav2Lip repository directory")
    args = parser.parse_args()

    run_lip_sync(
        face_video=args.input,
        audio_path=args.audio,
        output_path=args.output,
        checkpoint=args.checkpoint,
        wav2lip_dir=args.wav2lip_dir,
    )


if __name__ == "__main__":
    main()
