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
import tempfile


def dtw_align(orig_audio: np.ndarray, tts_audio: np.ndarray, sr: int) -> np.ndarray:
    """Warp tts_audio to match the timing of orig_audio using DTW on MFCCs."""
    orig_mfcc = librosa.feature.mfcc(y=orig_audio, sr=sr, n_mfcc=13)
    tts_mfcc = librosa.feature.mfcc(y=tts_audio, sr=sr, n_mfcc=13)
    _, wp = librosa.sequence.dtw(orig_mfcc, tts_mfcc, subseq=True)

    # wp[:, 0] = orig frame indices, wp[:, 1] = tts frame indices
    # The path is returned in reverse order by librosa; reverse it
    orig_frames = wp[::-1, 0]
    tts_frames = wp[::-1, 1]

    # Convert frame indices to sample indices
    hop_length = 512  # librosa default
    orig_samples = librosa.frames_to_samples(orig_frames, hop_length=hop_length)
    tts_samples = librosa.frames_to_samples(tts_frames, hop_length=hop_length)

    # Clamp to valid range
    tts_samples = np.clip(tts_samples, 0, len(tts_audio) - 1)
    orig_samples = np.clip(orig_samples, 0, len(orig_audio) - 1)

    # Target positions: evenly spaced over the orig audio length
    target_positions = np.arange(len(orig_audio))

    # Interpolate tts sample positions for each target (orig) sample position
    source_positions = np.interp(target_positions, orig_samples, tts_samples.astype(float))

    # Reconstruct warped audio by sampling tts_audio at the mapped positions
    warped_audio = np.interp(source_positions, np.arange(len(tts_audio)), tts_audio)
    return warped_audio.astype(np.float32)


def time_stretch_segment(audio_path: str, target_duration: float,
                         original_audio_path: str = None) -> tuple:
    audio, sr = librosa.load(audio_path, sr=22050)
    current_duration = len(audio) / sr

    if current_duration < 0.01:
        return audio, sr

    # Attempt DTW alignment if original audio is available
    if original_audio_path and Path(original_audio_path).exists():
        try:
            orig_audio, _ = librosa.load(original_audio_path, sr=sr)
            warped = dtw_align(orig_audio, audio, sr)
            ratio = len(warped) / max(len(audio), 1)
            if 0.6 <= ratio <= 1.5:
                return warped, sr
            # Ratio out of bounds — fall through to uniform stretch
        except Exception:
            pass  # Fall through to uniform stretch

    stretch_rate = current_duration / target_duration
    stretch_rate = float(np.clip(stretch_rate, 0.65, 1.5))

    stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
    return stretched, sr


def build_final_audio(segments_json: str, bgm_path: str,
                      output_path: str, total_duration: float,
                      prosody_transfer: bool = False):
    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    # Lazy import only when prosody transfer is requested
    if prosody_transfer:
        from scripts.inference.prosody_transfer import transfer_prosody

    sample_rate = 22050
    total_samples = int(total_duration * sample_rate)
    final_audio = np.zeros(total_samples)

    for seg in segments:
        if 'audio_file' not in seg:
            continue
        # Normalize Windows-style backslash paths to forward slashes
        audio_file = Path(seg['audio_file'].replace('\\', '/'))
        if not audio_file.exists():
            continue
        seg = dict(seg)
        seg['audio_file'] = str(audio_file)

        target_duration = seg['end'] - seg['start']
        if target_duration < 0.05:
            continue

        start_sample = int(seg['start'] * sample_rate)
        orig_audio_file = seg.get('original_audio_file')
        stretched, sr = time_stretch_segment(seg['audio_file'], target_duration,
                                             original_audio_path=orig_audio_file)

        # Optional prosody transfer: apply original Japanese pitch contour
        if prosody_transfer:
            orig_audio_file = seg.get('original_audio_file')
            if orig_audio_file:
                orig_path = Path(orig_audio_file.replace('\\', '/'))
                if orig_path.exists():
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp_path = tmp.name
                        sf.write(tmp_path, stretched, sr)
                        transfer_prosody(str(orig_path), tmp_path, tmp_path, sr=sr)
                        stretched, _ = librosa.load(tmp_path, sr=sr)
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[Prosody] Skipping segment (error: {e})")

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
    parser.add_argument("--prosody-transfer", action="store_true",
                        help="Apply prosody transfer from original Japanese audio after time-stretching")
    args = parser.parse_args()

    build_final_audio(args.segments, args.bgm, args.output, args.duration,
                      prosody_transfer=args.prosody_transfer)
