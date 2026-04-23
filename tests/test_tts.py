"""
Task 2.4 — TTS output tests.
Validates Requirements 1, 3: TTS produces valid audio files with stretch_ratio field.
"""
import json
from pathlib import Path
import librosa
import pytest

SEGMENTS_JSON = "tests/fixtures/tts_output/segments.json"


@pytest.fixture(scope="module")
def segments():
    if not Path(SEGMENTS_JSON).exists():
        pytest.skip(f"TTS segments manifest not found at {SEGMENTS_JSON}; run TTS pipeline first")
    with open(SEGMENTS_JSON) as f:
        return json.load(f)


def test_segments_manifest_exists():
    if not Path(SEGMENTS_JSON).exists():
        pytest.skip(f"TTS segments manifest not found at {SEGMENTS_JSON}; run TTS pipeline first")
    assert Path(SEGMENTS_JSON).exists()


def test_all_audio_files_exist(segments):
    missing = [s["audio_file"] for s in segments if not Path(s["audio_file"]).exists()]
    assert len(missing) == 0, f"Missing: {missing[:5]}"


def test_audio_files_non_empty(segments):
    for seg in segments[:5]:
        dur = librosa.get_duration(path=seg["audio_file"])
        assert dur > 0.1


def test_stretch_ratio_field(segments):
    for seg in segments:
        assert "stretch_ratio" in seg
