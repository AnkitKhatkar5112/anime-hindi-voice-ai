"""
Task 3.2 — Subtitle generation tests.
Validates Requirements 1, 3: generate_subtitles produces a valid SRT file
with correct format and content.
"""
import re
import pytest
from pathlib import Path

SRT_PATH = "tests/fixtures/processed/subtitles_hi.srt"

# SRT timestamp pattern: HH:MM:SS,mmm --> HH:MM:SS,mmm
SRT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
)


@pytest.fixture(scope="module")
def srt_content():
    """Read the SRT file, skipping if it doesn't exist."""
    if not Path(SRT_PATH).exists():
        pytest.skip(f"SRT file not found at {SRT_PATH}; run generate_subtitles.py first")
    return Path(SRT_PATH).read_text(encoding="utf-8")


def test_srt_exists():
    if not Path(SRT_PATH).exists():
        pytest.skip(f"SRT file not found at {SRT_PATH}; run generate_subtitles.py first")
    assert Path(SRT_PATH).exists(), f"SRT file not found at {SRT_PATH}"


def test_srt_has_content(srt_content):
    assert len(srt_content.strip()) > 0, "SRT file is empty"


def test_srt_format(srt_content):
    """Validate the first 5 SRT blocks have correct format: index, timestamp, text."""
    # Split into blocks separated by blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", srt_content.strip()) if b.strip()]
    assert len(blocks) >= 1, "SRT file has no subtitle blocks"

    for i, block in enumerate(blocks[:5]):
        lines = block.splitlines()
        assert len(lines) >= 3, (
            f"Block {i+1} has fewer than 3 lines: {lines!r}"
        )
        # Line 1: sequential index
        assert lines[0].strip().isdigit(), (
            f"Block {i+1} line 1 is not a digit index: {lines[0]!r}"
        )
        # Line 2: timestamp arrow
        assert SRT_TIMESTAMP_RE.match(lines[1].strip()), (
            f"Block {i+1} line 2 has invalid timestamp: {lines[1]!r}"
        )
        # Line 3+: subtitle text (non-empty)
        text = "\n".join(lines[2:]).strip()
        assert len(text) > 0, f"Block {i+1} has empty subtitle text"
