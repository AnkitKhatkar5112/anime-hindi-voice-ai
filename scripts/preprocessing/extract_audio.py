"""
Stage 1: Extract and preprocess audio from anime video files.
Handles noise reduction, normalization, and voice activity detection.
"""
import os
import argparse
import ffmpeg
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from pathlib import Path


def extract_audio(video_path: str, output_path: str, sample_rate: int = 22050) -> str:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(str(out_path), acodec='pcm_s16le', ac=1, ar=sample_rate)
        .overwrite_output()
        .run(quiet=True)
    )
    return str(out_path)


def preprocess_audio(audio_path: str, output_path: str, config: dict) -> str:
    audio, sr = librosa.load(audio_path, sr=config.get('sample_rate', 22050), mono=True)

    if config.get('noise_reduction', True):
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

    if config.get('normalize_volume', True):
        audio = librosa.util.normalize(audio)

    sf.write(output_path, audio, sr)
    print(f"[Preprocessing] Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video/audio file")
    parser.add_argument("--output", default="data/processed/audio.wav")
    args = parser.parse_args()

    raw_audio = extract_audio(args.input, "data/raw_audio/extracted.wav")
    preprocess_audio(raw_audio, args.output, {"noise_reduction": True, "normalize_volume": True})
