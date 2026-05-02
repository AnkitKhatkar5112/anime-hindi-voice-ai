"""
Stage 2: Audio source separation using Demucs.

Splits extracted audio into vocals and background (BGM + SFX) stems.
Computes SNR on the vocals stem and triggers ducking fallback if quality
is insufficient.

Inputs:  data/raw_audio/extracted.wav
Outputs: data/processed/vocals.wav
         data/processed/background.wav
         data/processed/separation_quality.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# SNR threshold below which ducking fallback is triggered
_SNR_THRESHOLD_DB = 10.0

# Target sample rate for all pipeline audio
_TARGET_SAMPLE_RATE = 22050

# Demucs model to use
_DEMUCS_MODEL = "htdemucs"

# Ducking volume levels for fallback mode (kept for backward compatibility)
_DUCKING_DIALOGUE_GAIN = 0.40   # volume during dialogue segments
_DUCKING_OUTSIDE_GAIN = 1.00    # volume outside dialogue segments


class SourceSeparator:
    """Separates audio into vocals and background stems using Demucs.

    If Demucs is not installed or fails, falls back to using the original
    audio for both stems. If the vocals SNR is below 10 dB, triggers
    ducking fallback mode.

    Class-level constants for ducking gain are exposed so the downstream
    mixer can apply them without hard-coding the values:

    - ``DUCKING_DIALOGUE_GAIN`` (0.40): background volume during dialogue.
    - ``DUCKING_OUTSIDE_GAIN``  (1.00): background volume outside dialogue.
    """

    # Public class-level ducking gain constants (Req 6 / task 5.4)
    DUCKING_DIALOGUE_GAIN: float = 0.40  # background volume during dialogue segments
    DUCKING_OUTSIDE_GAIN: float = 1.00   # background volume outside dialogue segments

    def __init__(self) -> None:
        self._demucs_available: bool | None = None  # lazily determined
        self.fallback_triggered: bool = False  # set to True when SNR < threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(
        self,
        input_wav: str,
        output_dir: str = "data/processed",
    ) -> Tuple[str, str]:
        """Run Demucs source separation on *input_wav*.

        Parameters
        ----------
        input_wav:
            Path to the input WAV file (e.g. ``data/raw_audio/extracted.wav``).
        output_dir:
            Directory where ``vocals.wav`` and ``background.wav`` will be
            written.  Defaults to ``data/processed``.

        Returns
        -------
        tuple[str, str]
            ``(vocals_path, background_path)`` — absolute or relative paths
            to the separated stems.  If Demucs is unavailable, both paths
            point to *input_wav*.
        """
        os.makedirs(output_dir, exist_ok=True)

        if not self._is_demucs_available():
            logger.warning(
                "[SourceSeparator] Demucs is not installed or not reachable. "
                "Falling back to using the original audio for both stems."
            )
            return (input_wav, input_wav)

        try:
            vocals_path, background_path = self._run_demucs(input_wav, output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[SourceSeparator] Demucs separation failed (%s: %s). "
                "Falling back to using the original audio for both stems.",
                type(exc).__name__,
                exc,
            )
            return (input_wav, input_wav)

        return (vocals_path, background_path)

    def compute_snr(self, vocals_path: str) -> float:
        """Compute the Signal-to-Noise Ratio of the vocals stem.

        SNR is estimated by treating the loudest 90% of short-time frames
        as signal and the quietest 10% as noise.

        Parameters
        ----------
        vocals_path:
            Path to the vocals WAV file.

        Returns
        -------
        float
            SNR in dB.  Returns 0.0 if the audio is silent.
        """
        audio, _ = librosa.load(vocals_path, sr=_TARGET_SAMPLE_RATE, mono=True)

        if audio.size == 0 or np.all(audio == 0):
            return 0.0

        # Compute RMS energy in short frames
        frame_length = 2048
        hop_length = 512
        rms_frames = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        if rms_frames.size == 0:
            return 0.0

        # Sort frames by energy; bottom 10% → noise estimate
        sorted_rms = np.sort(rms_frames)
        noise_cutoff = max(1, int(len(sorted_rms) * 0.10))
        noise_frames = sorted_rms[:noise_cutoff]
        signal_frames = sorted_rms[noise_cutoff:]

        noise_power = float(np.mean(noise_frames ** 2))
        signal_power = float(np.mean(signal_frames ** 2))

        if noise_power <= 0:
            # No noise detected — treat as very high SNR
            return 60.0

        if signal_power <= 0:
            return 0.0

        snr_db = 10.0 * math.log10(signal_power / noise_power)
        return snr_db

    def check_quality(
        self,
        vocals_path: str,
        input_wav: str,  # kept for API symmetry / future ducking use
    ) -> Tuple[bool, float]:
        """Check vocals quality and decide whether to trigger fallback.

        Parameters
        ----------
        vocals_path:
            Path to the separated vocals WAV.
        input_wav:
            Path to the original extracted audio (used if fallback is needed).

        Returns
        -------
        tuple[bool, float]
            ``(fallback_triggered, snr_db)``.
        """
        snr_db = self.compute_snr(vocals_path)
        fallback_triggered = snr_db < _SNR_THRESHOLD_DB

        if fallback_triggered:
            logger.warning(
                "[SourceSeparator] Low vocals SNR (%.1f dB < %.1f dB threshold) — "
                "ducking fallback mode activated. Background volume will be %.0f%% "
                "during dialogue and %.0f%% outside dialogue.",
                snr_db,
                _SNR_THRESHOLD_DB,
                _DUCKING_DIALOGUE_GAIN * 100,
                _DUCKING_OUTSIDE_GAIN * 100,
            )

        self.fallback_triggered = fallback_triggered
        return (fallback_triggered, snr_db)

    def write_quality_report(
        self,
        snr_db: float,
        fallback_triggered: bool,
        model: str = _DEMUCS_MODEL,
        output_dir: str = "data/processed",
    ) -> str:
        """Write ``separation_quality.json`` to *output_dir*.

        Parameters
        ----------
        snr_db:
            Computed SNR value in dB.
        fallback_triggered:
            Whether the ducking fallback was activated.
        model:
            Name of the Demucs model used (or ``"none"`` for fallback).
        output_dir:
            Directory where the JSON file will be written.

        Returns
        -------
        str
            Path to the written JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)
        report = {
            "snr_db": round(snr_db, 4),
            "fallback_triggered": fallback_triggered,
            "model": model,
        }
        out_path = os.path.join(output_dir, "separation_quality.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("[SourceSeparator] Quality report written to %s", out_path)
        return out_path

    def run(
        self,
        input_wav: str,
        output_dir: str = "data/processed",
    ) -> Tuple[str, str, bool]:
        """Orchestrate the full separation pipeline.

        Runs: separate → check quality → write report.

        Parameters
        ----------
        input_wav:
            Path to the input WAV file.
        output_dir:
            Directory for output files.

        Returns
        -------
        tuple[str, str, bool]
            ``(vocals_path, background_path, fallback_triggered)``.

            If Demucs is unavailable *or* SNR < 10 dB, both paths point to
            *input_wav* and ``fallback_triggered`` is ``True``.
        """
        vocals_path, background_path = self.separate(input_wav, output_dir)

        # If separation fell back (both paths == input_wav), skip SNR check
        demucs_succeeded = vocals_path != input_wav

        if demucs_succeeded:
            fallback_triggered, snr_db = self.check_quality(vocals_path, input_wav)
            model = _DEMUCS_MODEL
            if fallback_triggered:
                # Use original audio for both stems in ducking mode
                vocals_path = input_wav
                background_path = input_wav
        else:
            fallback_triggered = True
            snr_db = 0.0
            model = "none"
            self.fallback_triggered = True
            logger.warning(
                "[SourceSeparator] Demucs unavailable — ducking fallback mode activated. "
                "Background volume will be %.0f%% during dialogue and %.0f%% outside dialogue.",
                self.DUCKING_DIALOGUE_GAIN * 100,
                self.DUCKING_OUTSIDE_GAIN * 100,
            )

        self.write_quality_report(snr_db, fallback_triggered, model=model, output_dir=output_dir)

        return (vocals_path, background_path, fallback_triggered)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_demucs_available(self) -> bool:
        """Return True if Demucs can be imported or invoked."""
        if self._demucs_available is not None:
            return self._demucs_available

        # Try Python import first
        try:
            import demucs  # noqa: F401
            self._demucs_available = True
            return True
        except ImportError:
            pass

        # Try subprocess invocation as fallback
        try:
            result = subprocess.run(
                ["python", "-m", "demucs", "--help"],
                capture_output=True,
                timeout=10,
            )
            self._demucs_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._demucs_available = False

        return self._demucs_available

    def _run_demucs(self, input_wav: str, output_dir: str) -> Tuple[str, str]:
        """Run Demucs and copy stems to *output_dir*.

        Parameters
        ----------
        input_wav:
            Path to the input WAV file.
        output_dir:
            Destination directory for ``vocals.wav`` and ``background.wav``.

        Returns
        -------
        tuple[str, str]
            ``(vocals_path, background_path)`` inside *output_dir*.

        Raises
        ------
        RuntimeError
            If Demucs exits with a non-zero return code.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                "python", "-m", "demucs",
                "--two-stems", "vocals",
                "--name", _DEMUCS_MODEL,
                "--out", tmp_dir,
                input_wav,
            ]
            logger.info("[SourceSeparator] Running Demucs: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Demucs exited with code {result.returncode}.\n"
                    f"stderr: {result.stderr}"
                )

            # Demucs writes to: {tmp_dir}/{model}/{track_name}/vocals.wav
            #                                                  /no_vocals.wav
            input_stem = Path(input_wav).stem
            demucs_out = Path(tmp_dir) / _DEMUCS_MODEL / input_stem

            raw_vocals = demucs_out / "vocals.wav"
            raw_background = demucs_out / "no_vocals.wav"

            if not raw_vocals.exists() or not raw_background.exists():
                raise RuntimeError(
                    f"Expected Demucs output not found in {demucs_out}. "
                    f"Contents: {list(demucs_out.iterdir()) if demucs_out.exists() else 'directory missing'}"
                )

            vocals_path = os.path.join(output_dir, "vocals.wav")
            background_path = os.path.join(output_dir, "background.wav")

            self._resample_and_save(str(raw_vocals), vocals_path)
            self._resample_and_save(str(raw_background), background_path)

        logger.info(
            "[SourceSeparator] Separation complete. vocals=%s background=%s",
            vocals_path,
            background_path,
        )
        return (vocals_path, background_path)

    def _resample_and_save(self, src: str, dst: str) -> None:
        """Load *src*, resample to 22050 Hz, and write to *dst*."""
        audio, sr = librosa.load(src, sr=_TARGET_SAMPLE_RATE, mono=True)
        sf.write(dst, audio, _TARGET_SAMPLE_RATE)
        logger.debug(
            "[SourceSeparator] Resampled %s → %s (sr=%d)", src, dst, _TARGET_SAMPLE_RATE
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Separate audio into vocals and background stems."
    )
    parser.add_argument(
        "--input",
        default="data/raw_audio/extracted.wav",
        help="Input WAV file (default: data/raw_audio/extracted.wav)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    args = parser.parse_args()

    separator = SourceSeparator()
    vocals, background, fallback = separator.run(args.input, args.output_dir)
    print(f"vocals:     {vocals}")
    print(f"background: {background}")
    print(f"fallback:   {fallback}")
