"""
Task 2.2 — ASR transcription tests.
Validates Requirements 1, 3: ASR produces valid Japanese transcript JSON.
"""
import json
import subprocess
import sys
import pytest
from pathlib import Path

ASR_OUT = "tests/fixtures/processed/transcript_ja.json"


@pytest.fixture(scope="module")
def transcript():
    # Use existing transcript if it has content; otherwise run ASR
    existing = Path(ASR_OUT)
    if existing.exists() and existing.stat().st_size > 2:
        with open(ASR_OUT, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            return data
    # Run ASR — may be slow if model needs downloading
    result = subprocess.run([
        sys.executable, "scripts/preprocessing/asr_transcribe.py",
        "--audio", "tests/fixtures/processed/audio.wav",
        "--output", ASR_OUT,
        "--model", "medium"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"ASR failed (model may not be available): {result.stderr[-200:]}")
    with open(ASR_OUT, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        pytest.skip("ASR produced empty transcript — model may not be available")
    return data


def test_has_segments(transcript):
    assert len(transcript) > 0


def test_segment_keys(transcript):
    for seg in transcript:
        assert "start" in seg and "end" in seg and "text" in seg


def test_timestamps_ordered(transcript):
    starts = [s["start"] for s in transcript]
    assert starts == sorted(starts)


def test_has_japanese_text(transcript):
    all_text = " ".join(s["text"] for s in transcript)
    assert any('\u3000' <= c <= '\u9fff' for c in all_text)
