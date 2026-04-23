"""
Task 2.1 — Audio extraction tests.
Validates Requirements 1, 3: extract_audio produces a valid 22050 Hz mono WAV.
"""
import librosa
from pathlib import Path

AUDIO_OUT = "tests/fixtures/processed/audio.wav"


def test_output_exists():
    assert Path(AUDIO_OUT).exists()


def test_sample_rate():
    _, sr = librosa.load(AUDIO_OUT, sr=None)
    assert sr == 22050


def test_mono():
    audio, _ = librosa.load(AUDIO_OUT, mono=False, sr=None)
    assert audio.ndim == 1


def test_non_empty():
    audio, sr = librosa.load(AUDIO_OUT, sr=None)
    assert len(audio) / sr > 1.0


def test_duration_reasonable():
    duration = librosa.get_duration(path=AUDIO_OUT)
    assert 20 < duration < 40
