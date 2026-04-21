"""
Task 2.1 — Verify extract_audio.py preprocessing on a synthetic audio clip.
Validates Requirements 3: audio extraction produces 22050 Hz mono WAV.
"""
import sys
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocessing.extract_audio import preprocess_audio

SAMPLE_RATE = 22050
DURATION_SEC = 5  # seconds
FIXTURE_WAV = "tests/fixtures/synthetic_input.wav"
OUTPUT_WAV = "data/processed/audio.wav"


def create_synthetic_wav(path: str, duration: float = DURATION_SEC, sr: int = SAMPLE_RATE):
    """Create a simple sine-wave WAV for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix of 440 Hz and 880 Hz tones to simulate voice-like content
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)
    return path


def test_preprocess_audio_produces_correct_properties():
    # 1. Create synthetic input
    create_synthetic_wav(FIXTURE_WAV)
    assert Path(FIXTURE_WAV).exists(), "Fixture WAV was not created"

    # 2. Run preprocessing
    Path(OUTPUT_WAV).parent.mkdir(parents=True, exist_ok=True)
    result_path = preprocess_audio(
        FIXTURE_WAV,
        OUTPUT_WAV,
        {"noise_reduction": True, "normalize_volume": True, "sample_rate": SAMPLE_RATE},
    )

    # 3. Verify output file exists
    assert Path(result_path).exists(), f"Output WAV not found at {result_path}"

    # 4. Load and check properties
    audio, sr = librosa.load(result_path, sr=None, mono=False)
    assert sr == SAMPLE_RATE, f"Expected sample rate 22050, got {sr}"

    # mono check: shape should be 1-D (librosa returns (samples,) for mono)
    assert audio.ndim == 1, f"Expected mono (1-D), got shape {audio.shape}"

    # 5. Duration check
    duration = librosa.get_duration(y=audio, sr=sr)
    print(f"\n[Task 2.1] Output WAV properties:")
    print(f"  Path:        {result_path}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels:    mono (1)")
    print(f"  Duration:    {duration:.3f}s  (source: {DURATION_SEC}s)")

    # Duration should be within 1% of source
    assert abs(duration - DURATION_SEC) / DURATION_SEC < 0.01, (
        f"Duration mismatch: expected ~{DURATION_SEC}s, got {duration:.3f}s"
    )

    print("[Task 2.1] All checks passed ✓")


if __name__ == "__main__":
    test_preprocess_audio_produces_correct_properties()
