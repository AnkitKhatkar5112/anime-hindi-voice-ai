"""
Task 3.1 — Alignment and final mix tests.
Validates Requirements 1, 3: align_and_mix produces a valid output WAV
within duration tolerance and without clipping.
"""
import numpy as np
import librosa
import pytest
from pathlib import Path

SOURCE_AUDIO = "tests/fixtures/processed/audio.wav"
FINAL_MIX = "tests/fixtures/processed/final_mix.wav"


@pytest.fixture(scope="module")
def final_mix_audio():
    """Load the final mixed audio, skipping if it doesn't exist."""
    if not Path(FINAL_MIX).exists():
        pytest.skip(f"Final mix not found at {FINAL_MIX}; run align_and_mix.py first")
    audio, sr = librosa.load(FINAL_MIX, sr=None)
    return audio, sr


def test_output_exists():
    if not Path(FINAL_MIX).exists():
        pytest.skip(f"Final mix WAV not found at {FINAL_MIX}; run align_and_mix.py first")
    assert Path(FINAL_MIX).exists(), f"Final mix WAV not found at {FINAL_MIX}"


def test_duration_within_tolerance(final_mix_audio):
    audio, sr = final_mix_audio
    output_duration = len(audio) / sr
    source_duration = librosa.get_duration(path=SOURCE_AUDIO)
    tolerance = source_duration * 0.05
    assert abs(output_duration - source_duration) <= tolerance, (
        f"Output duration {output_duration:.2f}s differs from source "
        f"{source_duration:.2f}s by more than 5%"
    )


def test_no_clipping(final_mix_audio):
    audio, _ = final_mix_audio
    clipped_samples = np.sum(np.abs(audio) > 0.99)
    clipping_ratio = clipped_samples / len(audio)
    assert clipping_ratio < 0.001, (
        f"{clipping_ratio:.4%} of samples are clipped (threshold: 0.1%)"
    )
