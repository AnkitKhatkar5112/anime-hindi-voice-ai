"""
Stage 1 – Ingest: SRT subtitle file parser.

Reads ``.srt`` subtitle files and converts them into timed :class:`Segment`
objects for use in the production dubbing pipeline.  Supports both UTF-8 and
UTF-8-BOM encoded files and applies a configurable alignment-correction offset
(``--subtitle-offset``, default 0 ms).

See Requirement 2 (SRT Subtitle Input Path) in the production-dub-pipeline
spec for the full acceptance criteria.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from scripts.inference.models import Segment

logger = logging.getLogger(__name__)


class SRTParser:
    """Parse ``.srt`` subtitle files into pipeline :class:`Segment` objects.

    Implements **Requirement 2 – SRT Subtitle Input Path** from the
    production-dub-pipeline spec:

    * Reads ``.srt`` files encoded as UTF-8 or UTF-8-BOM.
    * Converts each subtitle entry into a :class:`Segment` with ``start``/
      ``end`` times (in seconds) and ``source_text`` populated.
    * Applies a configurable alignment-correction *offset* (milliseconds,
      default 0) to every timestamp.
    * Logs the applied offset and the number of segments produced.
    * Returns an empty list (triggering Whisper ASR fallback) when no valid
      entries are found.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, path: str, offset_ms: int = 0) -> List[Segment]:
        """Read an SRT file and return a list of :class:`Segment` objects.

        Parameters
        ----------
        path:
            Filesystem path to the ``.srt`` file.
        offset_ms:
            Alignment-correction offset in milliseconds applied to every
            ``start`` and ``end`` timestamp.  May be negative.  Defaults to
            ``0`` (no correction).

        Returns
        -------
        List[Segment]
            Parsed segments with timestamps adjusted by *offset_ms*.  Returns
            an empty list if the file contains no valid SRT entries (the
            caller should fall back to Whisper ASR in that case).

        Raises
        ------
        FileNotFoundError
            If *path* does not exist on disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SRT file not found: {path}")

        # Read file with UTF-8-BOM support
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        # Parse SRT blocks
        segments = []
        # Split by double newlines to separate blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # First line: index
            try:
                index = int(lines[0].strip())
            except ValueError:
                continue
            
            # Second line: timecode (HH:MM:SS,mmm --> HH:MM:SS,mmm)
            timecode_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
                lines[1].strip()
            )
            if not timecode_match:
                continue
            
            # Parse start time
            h1, m1, s1, ms1 = map(int, timecode_match.groups()[:4])
            start_sec = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            
            # Parse end time
            h2, m2, s2, ms2 = map(int, timecode_match.groups()[4:])
            end_sec = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            
            # Apply offset and clamp to >= 0.0
            offset_sec = offset_ms / 1000.0
            raw_start = start_sec + offset_sec
            raw_end = end_sec + offset_sec
            clamped_start = max(0.0, raw_start)
            clamped_end = max(0.0, raw_end)

            # Track whether this segment was clamped (negative before clamp)
            was_clamped = raw_start < 0.0 or raw_end < 0.0

            # Remaining lines: text content
            text_lines = [line.strip() for line in lines[2:]]
            source_text = '\n'.join(text_lines)

            # Create segment
            segment = Segment(
                segment_id=f"seg_{index:03d}",
                start=clamped_start,
                end=clamped_end,
                speaker_id="",
                source_text=source_text
            )
            segments.append((segment, was_clamped))

        # Separate segments from clamping flags
        result_segments = [seg for seg, _ in segments]
        clamped_count = sum(1 for _, clamped in segments if clamped)

        # Log results
        if len(result_segments) == 0:
            logger.error(
                f"[SRT] Zero valid entries parsed from {path} — falling back to Whisper ASR"
            )
            return []

        if offset_ms != 0:
            adjusted_count = len(result_segments)
            logger.info(
                f"[SRTParser] Applied offset {offset_ms}ms to {adjusted_count} segments"
                f" ({clamped_count} clamped to 0ms)"
            )
        else:
            logger.info(f"[SRTParser] Parsed {len(result_segments)} segments (offset: 0ms)")

        return result_segments

    def serialize(self, segments: List[Segment]) -> str:
        """Produce a valid SRT string from a list of :class:`Segment` objects.

        Parameters
        ----------
        segments:
            Ordered list of segments to serialise.  Each segment's
            ``source_text``, ``start``, and ``end`` fields are used.

        Returns
        -------
        str
            A well-formed SRT document as a Unicode string, suitable for
            writing directly to a ``.srt`` file.
        """
        raise NotImplementedError
