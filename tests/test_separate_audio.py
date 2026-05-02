"""
Task 5.7 — Unit tests for SourceSeparator.

Validates Req 6.5: if Demucs is not installed, SourceSeparator logs a warning
and returns (input_wav, input_wav) without raising an exception.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from scripts.preprocessing.separate_audio import SourceSeparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sine_wav(path: str, duration_s: float = 2.0, sr: int = 22050) -> str:
    """Write a simple sine-wave WAV to *path* and return the path."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, audio, sr)
    return path


def _write_silent_wav(path: str, duration_s: float = 2.0, sr: int = 22050) -> str:
    """Write a silent WAV to *path* and return the path."""
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(path, audio, sr)
    return path


# ---------------------------------------------------------------------------
# Task 5.7 — Demucs not installed: graceful fallback
# ---------------------------------------------------------------------------

class TestDemucsNotInstalled:
    """Req 6.5: if Demucs is not installed, log warning and return (input, input)."""

    def test_separate_returns_input_wav_for_both_stems(self, tmp_path, caplog):
        """separate() returns (input_wav, input_wav) when Demucs is unavailable."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()

        # Simulate Demucs not being importable AND subprocess failing
        with patch.object(separator, "_is_demucs_available", return_value=False):
            with caplog.at_level(logging.WARNING, logger="scripts.preprocessing.separate_audio"):
                vocals, background = separator.separate(input_wav, str(tmp_path))

        assert vocals == input_wav, "vocals path should equal input_wav on fallback"
        assert background == input_wav, "background path should equal input_wav on fallback"

    def test_separate_does_not_raise_when_demucs_missing(self, tmp_path):
        """separate() must not raise any exception when Demucs is unavailable."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()

        with patch.object(separator, "_is_demucs_available", return_value=False):
            try:
                vocals, background = separator.separate(input_wav, str(tmp_path))
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"separate() raised an unexpected exception: {exc}")

    def test_separate_logs_warning_when_demucs_missing(self, tmp_path, caplog):
        """separate() must emit a WARNING-level log when Demucs is unavailable."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()

        with patch.object(separator, "_is_demucs_available", return_value=False):
            with caplog.at_level(logging.WARNING, logger="scripts.preprocessing.separate_audio"):
                separator.separate(input_wav, str(tmp_path))

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages, "Expected at least one WARNING log when Demucs is missing"

    def test_separate_import_error_via_patch(self, tmp_path, caplog):
        """Simulate ImportError for demucs module via builtins.__import__ patch."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "demucs":
                raise ImportError("No module named 'demucs'")
            return real_import(name, *args, **kwargs)

        separator = SourceSeparator()

        # Also patch subprocess so the fallback check also fails
        with patch("builtins.__import__", side_effect=mock_import):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                # Reset cached availability so it re-checks
                separator._demucs_available = None

                with caplog.at_level(logging.WARNING, logger="scripts.preprocessing.separate_audio"):
                    vocals, background = separator.separate(input_wav, str(tmp_path))

        assert vocals == input_wav
        assert background == input_wav


# ---------------------------------------------------------------------------
# compute_snr — unit tests
# ---------------------------------------------------------------------------

class TestComputeSNR:
    """Tests for SourceSeparator.compute_snr()."""

    def test_snr_positive_for_tonal_signal(self, tmp_path):
        """A clean sine wave should have a positive SNR."""
        wav_path = str(tmp_path / "vocals.wav")
        _write_sine_wav(wav_path, duration_s=3.0)

        separator = SourceSeparator()
        snr = separator.compute_snr(wav_path)

        assert snr > 0.0, f"Expected positive SNR for a clean sine wave, got {snr}"

    def test_snr_returns_zero_for_silent_audio(self, tmp_path):
        """Silent audio should return SNR of 0.0."""
        wav_path = str(tmp_path / "silent.wav")
        _write_silent_wav(wav_path)

        separator = SourceSeparator()
        snr = separator.compute_snr(wav_path)

        assert snr == 0.0, f"Expected 0.0 SNR for silent audio, got {snr}"

    def test_snr_is_float(self, tmp_path):
        """compute_snr() must return a float."""
        wav_path = str(tmp_path / "vocals.wav")
        _write_sine_wav(wav_path)

        separator = SourceSeparator()
        snr = separator.compute_snr(wav_path)

        assert isinstance(snr, float)


# ---------------------------------------------------------------------------
# check_quality — unit tests
# ---------------------------------------------------------------------------

class TestCheckQuality:
    """Tests for SourceSeparator.check_quality()."""

    def test_no_fallback_for_high_snr(self, tmp_path):
        """A clean signal should not trigger fallback (SNR >= 10 dB)."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path, duration_s=3.0)
        _write_sine_wav(input_wav, duration_s=3.0)

        separator = SourceSeparator()
        # Patch compute_snr to return a known high value
        with patch.object(separator, "compute_snr", return_value=20.0):
            fallback, snr = separator.check_quality(vocals_path, input_wav)

        assert fallback is False
        assert snr == 20.0

    def test_fallback_triggered_for_low_snr(self, tmp_path, caplog):
        """SNR below 10 dB must trigger fallback and log a warning."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path)
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "compute_snr", return_value=5.0):
            with caplog.at_level(logging.WARNING, logger="scripts.preprocessing.separate_audio"):
                fallback, snr = separator.check_quality(vocals_path, input_wav)

        assert fallback is True
        assert snr == 5.0
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages, "Expected a WARNING log when SNR is below threshold"

    def test_fallback_at_exact_threshold(self, tmp_path):
        """SNR exactly at 10.0 dB should NOT trigger fallback (threshold is strict <)."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path)
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "compute_snr", return_value=10.0):
            fallback, snr = separator.check_quality(vocals_path, input_wav)

        assert fallback is False


# ---------------------------------------------------------------------------
# write_quality_report — unit tests
# ---------------------------------------------------------------------------

class TestWriteQualityReport:
    """Tests for SourceSeparator.write_quality_report()."""

    def test_report_file_created(self, tmp_path):
        """write_quality_report() must create separation_quality.json."""
        separator = SourceSeparator()
        report_path = separator.write_quality_report(
            snr_db=14.2,
            fallback_triggered=False,
            model="htdemucs",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(report_path)

    def test_report_schema(self, tmp_path):
        """separation_quality.json must contain snr_db, fallback_triggered, model."""
        separator = SourceSeparator()
        report_path = separator.write_quality_report(
            snr_db=14.2,
            fallback_triggered=False,
            model="htdemucs",
            output_dir=str(tmp_path),
        )
        with open(report_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert "snr_db" in data
        assert "fallback_triggered" in data
        assert "model" in data
        assert isinstance(data["snr_db"], float)
        assert isinstance(data["fallback_triggered"], bool)
        assert isinstance(data["model"], str)

    def test_report_values_match(self, tmp_path):
        """Report values must match the arguments passed."""
        separator = SourceSeparator()
        separator.write_quality_report(
            snr_db=7.5,
            fallback_triggered=True,
            model="htdemucs",
            output_dir=str(tmp_path),
        )
        with open(os.path.join(str(tmp_path), "separation_quality.json"), "r") as fh:
            data = json.load(fh)

        assert data["snr_db"] == pytest.approx(7.5, abs=0.01)
        assert data["fallback_triggered"] is True
        assert data["model"] == "htdemucs"


# ---------------------------------------------------------------------------
# run() — integration-style unit tests (Demucs mocked)
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for SourceSeparator.run() orchestration."""

    def test_run_fallback_when_demucs_unavailable(self, tmp_path):
        """run() returns (input_wav, input_wav, True) when Demucs is unavailable."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "_is_demucs_available", return_value=False):
            vocals, background, fallback = separator.run(input_wav, str(tmp_path))

        assert vocals == input_wav
        assert background == input_wav
        assert fallback is True

    def test_run_writes_quality_report(self, tmp_path):
        """run() must always write separation_quality.json."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "_is_demucs_available", return_value=False):
            separator.run(input_wav, str(tmp_path))

        report_path = os.path.join(str(tmp_path), "separation_quality.json")
        assert os.path.exists(report_path), "separation_quality.json must be written by run()"

    def test_run_fallback_snr_triggers_original_audio(self, tmp_path):
        """run() returns original audio paths when SNR check triggers fallback."""
        input_wav = str(tmp_path / "extracted.wav")
        vocals_wav = str(tmp_path / "vocals.wav")
        background_wav = str(tmp_path / "background.wav")
        _write_sine_wav(input_wav)
        _write_sine_wav(vocals_wav)
        _write_sine_wav(background_wav)

        separator = SourceSeparator()

        # Demucs "succeeds" but SNR is low
        with patch.object(separator, "_is_demucs_available", return_value=True):
            with patch.object(
                separator, "_run_demucs", return_value=(vocals_wav, background_wav)
            ):
                with patch.object(separator, "compute_snr", return_value=3.0):
                    vocals, background, fallback = separator.run(input_wav, str(tmp_path))

        assert fallback is True
        assert vocals == input_wav
        assert background == input_wav


# ---------------------------------------------------------------------------
# Task 5.4 — fallback_triggered attribute and ducking gain constants
# ---------------------------------------------------------------------------

class TestDuckingFallbackAttribute:
    """Task 5.4: fallback_triggered instance attribute and class-level gain constants."""

    def test_fallback_triggered_initialises_false(self):
        """fallback_triggered must be False on a fresh SourceSeparator instance."""
        separator = SourceSeparator()
        assert separator.fallback_triggered is False

    def test_fallback_triggered_set_true_after_low_snr(self, tmp_path):
        """check_quality() must set self.fallback_triggered=True when SNR < 10 dB."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path)
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "compute_snr", return_value=5.0):
            separator.check_quality(vocals_path, input_wav)

        assert separator.fallback_triggered is True

    def test_fallback_triggered_remains_false_for_high_snr(self, tmp_path):
        """check_quality() must leave self.fallback_triggered=False when SNR >= 10 dB."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path)
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "compute_snr", return_value=15.0):
            separator.check_quality(vocals_path, input_wav)

        assert separator.fallback_triggered is False

    def test_fallback_triggered_set_true_by_run_when_demucs_unavailable(self, tmp_path):
        """run() must set self.fallback_triggered=True when Demucs is unavailable."""
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "_is_demucs_available", return_value=False):
            separator.run(input_wav, str(tmp_path))

        assert separator.fallback_triggered is True

    def test_fallback_triggered_set_true_by_run_when_snr_low(self, tmp_path):
        """run() must set self.fallback_triggered=True when Demucs succeeds but SNR is low."""
        input_wav = str(tmp_path / "extracted.wav")
        vocals_wav = str(tmp_path / "vocals.wav")
        background_wav = str(tmp_path / "background.wav")
        _write_sine_wav(input_wav)
        _write_sine_wav(vocals_wav)
        _write_sine_wav(background_wav)

        separator = SourceSeparator()
        with patch.object(separator, "_is_demucs_available", return_value=True):
            with patch.object(
                separator, "_run_demucs", return_value=(vocals_wav, background_wav)
            ):
                with patch.object(separator, "compute_snr", return_value=3.0):
                    separator.run(input_wav, str(tmp_path))

        assert separator.fallback_triggered is True

    def test_ducking_dialogue_gain_class_constant(self):
        """DUCKING_DIALOGUE_GAIN class constant must be 0.40."""
        assert SourceSeparator.DUCKING_DIALOGUE_GAIN == pytest.approx(0.40)

    def test_ducking_outside_gain_class_constant(self):
        """DUCKING_OUTSIDE_GAIN class constant must be 1.00."""
        assert SourceSeparator.DUCKING_OUTSIDE_GAIN == pytest.approx(1.00)

    def test_ducking_gains_accessible_on_instance(self):
        """Ducking gain constants must be accessible on an instance."""
        separator = SourceSeparator()
        assert separator.DUCKING_DIALOGUE_GAIN == pytest.approx(0.40)
        assert separator.DUCKING_OUTSIDE_GAIN == pytest.approx(1.00)

    def test_warning_message_format_for_low_snr(self, tmp_path, caplog):
        """Warning message must contain SNR value and 'ducking' keyword."""
        vocals_path = str(tmp_path / "vocals.wav")
        input_wav = str(tmp_path / "extracted.wav")
        _write_sine_wav(vocals_path)
        _write_sine_wav(input_wav)

        separator = SourceSeparator()
        with patch.object(separator, "compute_snr", return_value=7.3):
            with caplog.at_level(logging.WARNING, logger="scripts.preprocessing.separate_audio"):
                separator.check_quality(vocals_path, input_wav)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages, "Expected at least one WARNING log"
        combined = " ".join(warning_messages)
        assert "7.3" in combined, "Warning must include the SNR value"
        assert "ducking" in combined.lower(), "Warning must mention 'ducking'"

    def test_fallback_background_is_full_audio_not_demucs_stem(self, tmp_path):
        """When SNR triggers fallback, background_path must be the full extracted audio."""
        input_wav = str(tmp_path / "extracted.wav")
        vocals_wav = str(tmp_path / "vocals.wav")
        background_wav = str(tmp_path / "background.wav")
        _write_sine_wav(input_wav)
        _write_sine_wav(vocals_wav)
        _write_sine_wav(background_wav)

        separator = SourceSeparator()
        with patch.object(separator, "_is_demucs_available", return_value=True):
            with patch.object(
                separator, "_run_demucs", return_value=(vocals_wav, background_wav)
            ):
                with patch.object(separator, "compute_snr", return_value=2.5):
                    _, returned_background, fallback = separator.run(input_wav, str(tmp_path))

        assert fallback is True
        assert returned_background == input_wav, (
            "In ducking fallback mode, background must be the full extracted audio, "
            f"not the Demucs stem. Got: {returned_background}"
        )
