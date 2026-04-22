"""
Prosody Transfer: Apply the pitch contour shape from an original Japanese
audio segment onto a Hindi TTS output using WORLD vocoder (pyworld).
"""
import numpy as np
import librosa
import soundfile as sf

try:
    import pyworld as pw
except ImportError:
    pw = None


def transfer_prosody(original_wav: str, tts_wav: str, output_wav: str, sr: int = 22050) -> None:
    """
    Transfer the F0 (pitch) contour shape from original_wav onto tts_wav
    and write the result to output_wav.

    Steps:
      1. Extract F0, spectral envelope, and aperiodicity from both signals.
      2. Normalize both F0 curves over voiced frames (F0 > 0) to zero-mean/unit-std.
      3. Resample the normalised original F0 to match the TTS frame count.
      4. Scale the resampled curve back using the TTS speaker's mean/std.
      5. Synthesize with the transferred F0 and TTS spectral features.

    Args:
        original_wav: Path to the original Japanese audio file.
        tts_wav:      Path to the Hindi TTS audio file.
        output_wav:   Destination path for the prosody-transferred audio.
        sr:           Sample rate used for loading and synthesis (default 22050).
    """
    if pw is None:
        raise ImportError(
            "pyworld is required for prosody transfer. "
            "Install it with: pip install pyworld"
        )

    orig, _ = librosa.load(original_wav, sr=sr)
    tts, _ = librosa.load(tts_wav, sr=sr)

    orig_f0, orig_sp, orig_ap = pw.wav2world(orig.astype(np.float64), sr)
    tts_f0, tts_sp, tts_ap = pw.wav2world(tts.astype(np.float64), sr)

    # --- 1. Compute stats over voiced frames only (F0 > 0) ---
    orig_voiced = orig_f0[orig_f0 > 0]
    tts_voiced = tts_f0[tts_f0 > 0]

    orig_mean = float(np.mean(orig_voiced)) if len(orig_voiced) > 0 else 1.0
    orig_std = float(np.std(orig_voiced)) if len(orig_voiced) > 1 else 1.0
    tts_mean = float(np.mean(tts_voiced)) if len(tts_voiced) > 0 else orig_mean
    tts_std = float(np.std(tts_voiced)) if len(tts_voiced) > 1 else orig_std

    # Avoid division by zero
    if orig_std < 1e-6:
        orig_std = 1.0
    if tts_std < 1e-6:
        tts_std = orig_std

    # --- 2. Normalise original F0 (voiced frames only) ---
    orig_f0_norm = np.where(orig_f0 > 0, (orig_f0 - orig_mean) / orig_std, 0.0)

    # --- 3. Resample normalised original F0 to TTS frame count ---
    tts_len = len(tts_f0)
    orig_len = len(orig_f0_norm)

    if orig_len != tts_len:
        orig_indices = np.linspace(0, orig_len - 1, tts_len)
        resampled_norm = np.interp(orig_indices, np.arange(orig_len), orig_f0_norm)
    else:
        resampled_norm = orig_f0_norm.copy()

    # --- 4. Scale to TTS speaker's mean/std, preserve unvoiced frames ---
    # Use TTS voiced mask to decide which frames stay voiced after transfer
    tts_voiced_mask = tts_f0 > 0
    transferred_f0 = np.where(
        tts_voiced_mask,
        resampled_norm * tts_std + tts_mean,
        0.0,
    )
    # Clip to a reasonable F0 range (50 Hz – 800 Hz) to avoid synthesis artefacts
    transferred_f0 = np.where(
        transferred_f0 > 0,
        np.clip(transferred_f0, 50.0, 800.0),
        0.0,
    )

    # --- 5. Synthesize ---
    out = pw.synthesize(transferred_f0, tts_sp, tts_ap, float(sr))
    sf.write(output_wav, out.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer pitch contour from an original audio file onto a TTS output."
    )
    parser.add_argument("--original", required=True, help="Path to original (Japanese) audio WAV")
    parser.add_argument("--tts", required=True, help="Path to Hindi TTS audio WAV")
    parser.add_argument("--output", required=True, help="Path for the prosody-transferred output WAV")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    args = parser.parse_args()

    transfer_prosody(args.original, args.tts, args.output, sr=args.sr)
    print(f"[Prosody] Transferred prosody saved to: {args.output}")
