"""
Stub replacement for run_pipeline.py — used for testing batch_process.py
without requiring ML models. Accepts the same CLI args, writes a dummy
output WAV to outputs/{stem}_{lang}_dub.wav, and exits 0.
"""
import argparse
import wave
import struct
import sys
from pathlib import Path


def write_silent_wav(path: str, duration_s: float = 0.5, sample_rate: int = 16000):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sample_rate * duration_s)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + 'h' * n_samples, *([0] * n_samples)))


def main():
    parser = argparse.ArgumentParser(description="Stub pipeline runner for testing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--speaker-wav", default=None)
    parser.add_argument("--skip-diarize", action="store_true")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--video-output", action="store_true")
    parser.add_argument("--face-video", default=None)
    args = parser.parse_args()

    stem = Path(args.input).stem
    output_path = f"outputs/{stem}_{args.lang}_dub.wav"
    write_silent_wav(output_path)
    print(f"[STUB] Wrote dummy output: {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
