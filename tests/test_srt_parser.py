"""
Unit tests for SRTParser – Task 4.3: offset_ms application and clamping.

Covers:
- Positive offset shifts all start/end times forward
- Negative offset shifts all start/end times backward
- Clamping: times never go below 0.0
- Zero offset leaves times unchanged
- FileNotFoundError on missing file
"""

from __future__ import annotations

import os
import tempfile

import pytest

from scripts.preprocessing.srt_parser import SRTParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_srt(content: str) -> str:
    """Write *content* to a temp .srt file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".srt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


SIMPLE_SRT = """\
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:05,500 --> 00:00:07,200
Second line
"""

EARLY_SRT = """\
1
00:00:00,100 --> 00:00:00,500
Very early segment

2
00:00:01,000 --> 00:00:02,000
Normal segment
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSRTParserOffset:
    """Tests for offset_ms application (Task 4.3, Req 2.7, 2.9)."""

    def test_zero_offset_leaves_times_unchanged(self):
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            segments = parser.parse(path, offset_ms=0)
            assert len(segments) == 2
            assert segments[0].start == pytest.approx(1.0)
            assert segments[0].end == pytest.approx(3.0)
            assert segments[1].start == pytest.approx(5.5)
            assert segments[1].end == pytest.approx(7.2)
        finally:
            os.unlink(path)

    def test_positive_offset_shifts_times_forward(self):
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            segments = parser.parse(path, offset_ms=500)  # +500ms = +0.5s
            assert segments[0].start == pytest.approx(1.5)
            assert segments[0].end == pytest.approx(3.5)
            assert segments[1].start == pytest.approx(6.0)
            assert segments[1].end == pytest.approx(7.7)
        finally:
            os.unlink(path)

    def test_negative_offset_shifts_times_backward(self):
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            segments = parser.parse(path, offset_ms=-500)  # -500ms = -0.5s
            assert segments[0].start == pytest.approx(0.5)
            assert segments[0].end == pytest.approx(2.5)
            assert segments[1].start == pytest.approx(5.0)
            assert segments[1].end == pytest.approx(6.7)
        finally:
            os.unlink(path)

    def test_negative_offset_clamps_start_to_zero(self):
        """A large negative offset must not produce negative start times."""
        path = _write_srt(EARLY_SRT)
        try:
            parser = SRTParser()
            # -300ms offset: first segment start=0.1s → 0.1-0.3=-0.2 → clamped to 0.0
            segments = parser.parse(path, offset_ms=-300)
            assert segments[0].start == pytest.approx(0.0), (
                "start time must be clamped to 0.0, not negative"
            )
        finally:
            os.unlink(path)

    def test_negative_offset_clamps_end_to_zero(self):
        """A very large negative offset must not produce negative end times."""
        path = _write_srt(EARLY_SRT)
        try:
            parser = SRTParser()
            # -600ms: first segment end=0.5s → 0.5-0.6=-0.1 → clamped to 0.0
            segments = parser.parse(path, offset_ms=-600)
            assert segments[0].end == pytest.approx(0.0), (
                "end time must be clamped to 0.0, not negative"
            )
        finally:
            os.unlink(path)

    def test_offset_applied_to_all_segments(self):
        """Every segment in the file must have the offset applied."""
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            offset_ms = 200
            segments_no_offset = parser.parse(path, offset_ms=0)
            segments_with_offset = parser.parse(path, offset_ms=offset_ms)

            offset_s = offset_ms / 1000.0
            for orig, shifted in zip(segments_no_offset, segments_with_offset):
                assert shifted.start == pytest.approx(orig.start + offset_s)
                assert shifted.end == pytest.approx(orig.end + offset_s)
        finally:
            os.unlink(path)

    def test_start_and_end_never_negative(self):
        """After any offset, start and end must always be >= 0.0."""
        path = _write_srt(EARLY_SRT)
        try:
            parser = SRTParser()
            # Extreme negative offset
            segments = parser.parse(path, offset_ms=-10000)
            for seg in segments:
                assert seg.start >= 0.0, f"start={seg.start} is negative"
                assert seg.end >= 0.0, f"end={seg.end} is negative"
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        parser = SRTParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.srt")

    def test_utf8_bom_file_parsed_correctly(self):
        """UTF-8-BOM encoded files must be parsed without errors."""
        fd, path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, "wb") as f:
            # Write UTF-8 BOM + content
            f.write(b"\xef\xbb\xbf")
            f.write(SIMPLE_SRT.encode("utf-8"))
        try:
            parser = SRTParser()
            segments = parser.parse(path, offset_ms=0)
            assert len(segments) == 2
            assert segments[0].start == pytest.approx(1.0)
        finally:
            os.unlink(path)

    def test_segment_ids_assigned(self):
        """Parsed segments must have segment_id set."""
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            segments = parser.parse(path)
            assert segments[0].segment_id == "seg_001"
            assert segments[1].segment_id == "seg_002"
        finally:
            os.unlink(path)

    def test_source_text_populated(self):
        """source_text must contain the subtitle text."""
        path = _write_srt(SIMPLE_SRT)
        try:
            parser = SRTParser()
            segments = parser.parse(path)
            assert segments[0].source_text == "Hello world"
            assert segments[1].source_text == "Second line"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests for Task 4.4: zero valid entries → error log + empty list returned
# ---------------------------------------------------------------------------

EMPTY_SRT = ""  # completely empty file

MALFORMED_SRT = """\
not a valid srt block at all
just some random text
"""

ONLY_WHITESPACE_SRT = "\n\n\n   \n\n"


class TestSRTParserZeroEntries:
    """Tests for zero-valid-entries guard (Task 4.4, Req 2 AC4)."""

    def test_empty_file_returns_empty_list(self):
        """An empty .srt file must return an empty list (not raise)."""
        path = _write_srt(EMPTY_SRT)
        try:
            parser = SRTParser()
            result = parser.parse(path)
            assert result == [], f"Expected [], got {result}"
        finally:
            os.unlink(path)

    def test_malformed_file_returns_empty_list(self):
        """A file with no valid SRT blocks must return an empty list."""
        path = _write_srt(MALFORMED_SRT)
        try:
            parser = SRTParser()
            result = parser.parse(path)
            assert result == [], f"Expected [], got {result}"
        finally:
            os.unlink(path)

    def test_whitespace_only_file_returns_empty_list(self):
        """A whitespace-only file must return an empty list."""
        path = _write_srt(ONLY_WHITESPACE_SRT)
        try:
            parser = SRTParser()
            result = parser.parse(path)
            assert result == [], f"Expected [], got {result}"
        finally:
            os.unlink(path)

    def test_zero_entries_logs_error(self, caplog):
        """Zero valid entries must produce an error-level log message."""
        import logging
        path = _write_srt(EMPTY_SRT)
        try:
            parser = SRTParser()
            with caplog.at_level(logging.ERROR, logger="scripts.preprocessing.srt_parser"):
                parser.parse(path)
            assert any(
                "Zero valid entries" in record.message and record.levelno == logging.ERROR
                for record in caplog.records
            ), f"Expected ERROR log with 'Zero valid entries', got: {caplog.records}"
        finally:
            os.unlink(path)

    def test_zero_entries_log_includes_path(self, caplog):
        """The error log message must include the file path."""
        import logging
        path = _write_srt(EMPTY_SRT)
        try:
            parser = SRTParser()
            with caplog.at_level(logging.ERROR, logger="scripts.preprocessing.srt_parser"):
                parser.parse(path)
            error_messages = [
                r.message for r in caplog.records if r.levelno == logging.ERROR
            ]
            assert any(path in msg for msg in error_messages), (
                f"Expected path '{path}' in error log, got: {error_messages}"
            )
        finally:
            os.unlink(path)

    def test_zero_entries_does_not_raise(self):
        """Zero valid entries must NOT raise an exception — caller checks for empty list."""
        path = _write_srt(EMPTY_SRT)
        try:
            parser = SRTParser()
            # Should complete without raising
            result = parser.parse(path)
            assert isinstance(result, list)
        finally:
            os.unlink(path)
