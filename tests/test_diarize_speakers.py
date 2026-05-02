"""
Task 6.7 — Unit tests for Voice_Embedding_Clusterer (fast-mode diarization).

Validates Req 4.7–4.9: when --fast-mode is set, Voice_Embedding_Clusterer
clusters segments by speaker using Resemblyzer embeddings and assigns
SPEAKER_XX IDs.

This test creates a synthetic 3-speaker audio fixture using sine waves at
distinct frequencies (200 Hz, 400 Hz, 800 Hz) to simulate different speakers,
then asserts that the clusterer produces exactly 3 distinct SPEAKER_XX IDs.

Both Resemblyzer and scikit-learn are mocked via unittest.mock.patch so the
test runs in CI without the full dependencies installed.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from scripts.preprocessing.diarize_speakers import Voice_Embedding_Clusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000  # Hz — matches Resemblyzer's expected input rate


def _sine_wave(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a mono float32 sine wave array."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _write_3speaker_wav(path: str, segment_duration_s: float = 2.0) -> str:
    """
    Write a WAV file with 3 consecutive segments, each simulating a different
    speaker via a distinct sine-wave frequency:
        Segment 0: 200 Hz  (speaker A)
        Segment 1: 400 Hz  (speaker B)
        Segment 2: 800 Hz  (speaker C)

    Returns the path for convenience.
    """
    seg_a = _sine_wave(200.0, segment_duration_s)
    seg_b = _sine_wave(400.0, segment_duration_s)
    seg_c = _sine_wave(800.0, segment_duration_s)
    audio = np.concatenate([seg_a, seg_b, seg_c])
    sf.write(path, audio, SAMPLE_RATE)
    return path


def _make_segments(segment_duration_s: float = 2.0) -> list:
    """
    Return 3 dict-based segments whose time ranges correspond to the 3
    sine-wave regions written by _write_3speaker_wav().
    """
    return [
        {"start": 0.0,                    "end": segment_duration_s},
        {"start": segment_duration_s,     "end": segment_duration_s * 2},
        {"start": segment_duration_s * 2, "end": segment_duration_s * 3},
    ]


# ---------------------------------------------------------------------------
# Fake embeddings — 3 clearly distinct unit vectors in 256-d space
# ---------------------------------------------------------------------------

def _make_fake_embeddings() -> list:
    """
    Return 3 orthogonal unit-vector embeddings that agglomerative clustering
    (cosine distance threshold 0.25) will always assign to 3 separate clusters.

    Cosine distance between any two of these vectors is 1.0 (orthogonal),
    which is well above the 0.25 threshold.
    """
    dim = 256
    e0 = np.zeros(dim, dtype=np.float32)
    e0[0] = 1.0

    e1 = np.zeros(dim, dtype=np.float32)
    e1[1] = 1.0

    e2 = np.zeros(dim, dtype=np.float32)
    e2[2] = 1.0

    return [e0, e1, e2]


# ---------------------------------------------------------------------------
# Fake AgglomerativeClustering that assigns one cluster per embedding
# ---------------------------------------------------------------------------

class _FakeAgglomerativeClustering:
    """
    Minimal stand-in for sklearn.cluster.AgglomerativeClustering.

    Assigns a unique label to each row in the input matrix, simulating the
    case where all embeddings are far apart (3 distinct speakers).
    """

    def __init__(self, **kwargs):
        self.labels_ = None

    def fit_predict(self, X):
        # Each row gets its own cluster label: 0, 1, 2, ...
        self.labels_ = np.arange(len(X))
        return self.labels_


def _make_sklearn_mock():
    """Build a sys.modules mock for sklearn that provides AgglomerativeClustering."""
    sklearn_mock = MagicMock()
    sklearn_cluster_mock = MagicMock()
    sklearn_cluster_mock.AgglomerativeClustering = _FakeAgglomerativeClustering
    sklearn_mock.cluster = sklearn_cluster_mock
    return sklearn_mock, sklearn_cluster_mock


def _make_resemblyzer_mock(fake_embeddings: list):
    """Build a sys.modules mock for resemblyzer with a VoiceEncoder stub."""
    mock_encoder = MagicMock()
    mock_encoder.embed_utterance.side_effect = list(fake_embeddings)
    mock_preprocess_wav = MagicMock(side_effect=lambda p: p)

    resemblyzer_mock = MagicMock()
    resemblyzer_mock.VoiceEncoder = MagicMock(return_value=mock_encoder)
    resemblyzer_mock.preprocess_wav = mock_preprocess_wav
    return resemblyzer_mock, mock_encoder


def _patched_modules(fake_embeddings: list) -> dict:
    """Return a sys.modules patch dict covering both resemblyzer and sklearn."""
    sklearn_mock, sklearn_cluster_mock = _make_sklearn_mock()
    resemblyzer_mock, _ = _make_resemblyzer_mock(fake_embeddings)
    return {
        "resemblyzer": resemblyzer_mock,
        "sklearn": sklearn_mock,
        "sklearn.cluster": sklearn_cluster_mock,
    }


# ---------------------------------------------------------------------------
# Task 6.7 — 3-speaker fixture produces 3 distinct SPEAKER_XX IDs
# ---------------------------------------------------------------------------

SPEAKER_PATTERN = re.compile(r"^SPEAKER_\d+$")


class TestVoiceEmbeddingClusterer3Speakers:
    """Req 4.7–4.9: fast-mode clustering assigns distinct SPEAKER_XX IDs."""

    def test_three_speakers_produce_three_distinct_ids_mocked(self, tmp_path):
        """
        With mocked Resemblyzer and sklearn, 3 segments with orthogonal
        embeddings must produce exactly 3 distinct SPEAKER_XX IDs.
        """
        wav_path = str(tmp_path / "3speakers.wav")
        _write_3speaker_wav(wav_path, segment_duration_s=2.0)

        segments = _make_segments(segment_duration_s=2.0)
        fake_embeddings = _make_fake_embeddings()

        with patch.dict("sys.modules", _patched_modules(fake_embeddings)):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster(segments, wav_path)

        # --- Assertions ---
        assert len(result) == 3, f"Expected 3 result segments, got {len(result)}"

        speaker_ids = [seg["speaker"] for seg in result]

        # All IDs must match SPEAKER_XX pattern
        for sid in speaker_ids:
            assert SPEAKER_PATTERN.match(sid), (
                f"Speaker ID '{sid}' does not match SPEAKER_XX pattern"
            )

        # Must be exactly 3 distinct IDs
        distinct_ids = set(speaker_ids)
        assert len(distinct_ids) == 3, (
            f"Expected 3 distinct SPEAKER_XX IDs, got {len(distinct_ids)}: {distinct_ids}"
        )

    def test_speaker_ids_match_pattern(self, tmp_path):
        """Every returned speaker ID must match the SPEAKER_\\d+ pattern."""
        wav_path = str(tmp_path / "3speakers.wav")
        _write_3speaker_wav(wav_path, segment_duration_s=2.0)

        segments = _make_segments(segment_duration_s=2.0)
        fake_embeddings = _make_fake_embeddings()

        with patch.dict("sys.modules", _patched_modules(fake_embeddings)):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster(segments, wav_path)

        for seg in result:
            sid = seg["speaker"]
            assert SPEAKER_PATTERN.match(sid), (
                f"Speaker ID '{sid}' does not match SPEAKER_XX pattern"
            )

    def test_result_length_equals_input_length(self, tmp_path):
        """cluster() must return the same number of segments as the input."""
        wav_path = str(tmp_path / "3speakers.wav")
        _write_3speaker_wav(wav_path, segment_duration_s=2.0)

        segments = _make_segments(segment_duration_s=2.0)
        fake_embeddings = _make_fake_embeddings()

        with patch.dict("sys.modules", _patched_modules(fake_embeddings)):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster(segments, wav_path)

        assert len(result) == len(segments)

    def test_segment_dataclass_objects_get_speaker_id_attribute(self, tmp_path):
        """
        cluster() also works with Segment dataclass objects (not just dicts).
        Each Segment must have its speaker_id attribute set to a SPEAKER_XX value.
        """
        from scripts.inference.models import Segment

        wav_path = str(tmp_path / "3speakers.wav")
        _write_3speaker_wav(wav_path, segment_duration_s=2.0)

        seg_duration = 2.0
        segments = [
            Segment(
                segment_id=f"seg_{i:03d}",
                start=i * seg_duration,
                end=(i + 1) * seg_duration,
                speaker_id="UNKNOWN",
                source_text="",
            )
            for i in range(3)
        ]

        fake_embeddings = _make_fake_embeddings()

        with patch.dict("sys.modules", _patched_modules(fake_embeddings)):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster(segments, wav_path)

        assert len(result) == 3
        speaker_ids = [seg.speaker_id for seg in result]
        distinct_ids = set(speaker_ids)

        for sid in speaker_ids:
            assert SPEAKER_PATTERN.match(sid), (
                f"Segment speaker_id '{sid}' does not match SPEAKER_XX pattern"
            )

        assert len(distinct_ids) == 3, (
            f"Expected 3 distinct SPEAKER_XX IDs for Segment objects, "
            f"got {len(distinct_ids)}: {distinct_ids}"
        )

    def test_empty_segments_returns_empty_list(self, tmp_path):
        """cluster() with an empty segment list must return an empty list."""
        wav_path = str(tmp_path / "3speakers.wav")
        _write_3speaker_wav(wav_path, segment_duration_s=2.0)

        with patch.dict("sys.modules", _patched_modules([])):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster([], wav_path)

        assert result == []

    def test_missing_wav_falls_back_to_speaker_00(self, tmp_path):
        """
        cluster() with a non-existent WAV path must fall back to SPEAKER_00
        for all segments without raising.
        """
        missing_wav = str(tmp_path / "does_not_exist.wav")
        segments = _make_segments(segment_duration_s=2.0)

        with patch.dict("sys.modules", _patched_modules([])):
            clusterer = Voice_Embedding_Clusterer(cache=None)
            result = clusterer.cluster(segments, missing_wav)

        assert len(result) == 3
        for seg in result:
            assert seg["speaker"] == "SPEAKER_00", (
                f"Expected SPEAKER_00 fallback, got '{seg['speaker']}'"
            )
