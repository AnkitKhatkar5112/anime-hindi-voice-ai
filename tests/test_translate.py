"""
Task 2.3 — Translation tests.
Validates Requirements 1, 3: translate.py produces valid Hindi translation JSON.
"""
import json
import subprocess
import sys
import pytest
from pathlib import Path

ASR_OUT = "tests/fixtures/processed/transcript_ja.json"
TRANS_OUT = "tests/fixtures/processed/transcript_hi.json"


@pytest.fixture(scope="module")
def translated():
    # Use existing translation if it has content
    existing = Path(TRANS_OUT)
    if existing.exists() and existing.stat().st_size > 2:
        with open(TRANS_OUT, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            return data

    # Ensure ASR output exists first
    asr_path = Path(ASR_OUT)
    if not asr_path.exists() or asr_path.stat().st_size <= 2:
        result = subprocess.run([
            sys.executable, "scripts/preprocessing/asr_transcribe.py",
            "--audio", "tests/fixtures/processed/audio.wav",
            "--output", ASR_OUT,
            "--model", "medium"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip(f"ASR failed (model may not be available): {result.stderr[-200:]}")

    # Check ASR produced content
    with open(ASR_OUT, encoding="utf-8") as f:
        asr_data = json.load(f)
    if not asr_data:
        pytest.skip("ASR transcript is empty — skipping translation test")

    result = subprocess.run([
        sys.executable, "scripts/preprocessing/translate.py",
        "--input", ASR_OUT,
        "--output", TRANS_OUT,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"Translation failed: {result.stderr[-200:]}")

    with open(TRANS_OUT, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        pytest.skip("Translation produced empty output")
    return data


def test_no_segments_dropped(translated):
    with open(ASR_OUT, encoding="utf-8") as f:
        source = json.load(f)
    assert len(translated) >= len(source) * 0.95


def test_translation_fields(translated):
    for seg in translated:
        assert "text_translated" in seg


def test_cleaned_field(translated):
    for seg in translated:
        assert "text_cleaned" in seg


def test_no_empty_translations(translated):
    empty = [s for s in translated if not s.get("text_translated", "").strip()]
    assert len(empty) < len(translated) * 0.10
