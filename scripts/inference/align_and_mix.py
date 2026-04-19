"""
Stage 5: Timing Alignment + Final Audio Mix
Time-stretches TTS segments to match original Japanese timing windows,
then mixes with background music and SFX.
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json


def time_stretch_segment(audio_path: str, target_duration: float) -> tuple:
    audio, sr = librosa.load(audio_path, sr=22050)
    current_duration = len(audio) / sr

    if current_duration < 0.01:
        return audio, sr

    stretch_rate = current_duration / target_duration
    stretch_rate = float(np.clip(stretch_rate, 0.65, 1.5))

    stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
    return stretched, sr


def build_final_audio(segments_json: str, bgm_path: str,
                      output_path: str, total_duration: float):
    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    sample_rate = 22050
    total_samples = int(total_duration * sample_rate)
    final_audio = np.zeros(total_samples)

    for seg in segments:
        if 'audio_file' not in seg or not Path(seg['audio_file']).exists():
            continue

        target_duration = seg['end'] - seg['start']
        if target_duration < 0.05:
            continue

        start_sample = int(seg['start'] * sample_rate)
        stretched, sr = time_stretch_segment(seg['audio_file'], target_duration)
        end_sample = min(start_sample + len(stretched), total_samples)
        actual_len = end_sample - start_sample

        final_audio[start_sample:end_sample] += stretched[:actual_len] * 0.85

    if bgm_path and Path(bgm_path).exists():
        bgm, _ = librosa.load(bgm_path, sr=sample_rate, duration=total_duration)
        bgm = librosa.util.fix_length(bgm, size=total_samples)
        final_audio += bgm * 0.20

    final_audio = np.clip(final_audio, -1.0, 1.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, final_audio, sample_rate)
    print(f"[Mixer] Final audio saved: {output_path} ({total_duration:.1f}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", required=True)
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--output", default="outputs/final_hindi_dub.wav")
    parser.add_argument("--duration", type=float, required=True, help="Total audio duration in seconds")
    args = parser.parse_args()

    build_final_audio(args.segments, args.bgm, args.output, args.duration)
