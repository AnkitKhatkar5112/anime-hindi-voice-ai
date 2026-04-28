"""
Unit tests for Segment, VoiceProfile, and VoiceMap data models.
Covers construction, default values, and field types.
"""

import pytest
from scripts.inference.models import Segment, VoiceProfile, VoiceMap, load_voice_map, save_voice_map


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------

class TestSegmentConstruction:
    def test_required_fields_only(self):
        seg = Segment(
            segment_id="seg_001",
            start=1.0,
            end=3.5,
            speaker_id="SPEAKER_00",
            source_text="こんにちは",
        )
        assert seg.segment_id == "seg_001"
        assert seg.start == 1.0
        assert seg.end == 3.5
        assert seg.speaker_id == "SPEAKER_00"
        assert seg.source_text == "こんにちは"

    def test_all_fields_explicit(self):
        seg = Segment(
            segment_id="seg_002",
            start=5.0,
            end=8.0,
            speaker_id="SPEAKER_01",
            source_text="何だと！",
            translated_text="What did you say!",
            dubbed_text="क्या बोला तूने!",
            emotion="angry",
            emotion_intensity=0.9,
            voice_profile="voice_male_deep",
            tts_audio_path="data/tts_output/seg_002.wav",
            stretch_ratio=1.2,
            discarded=False,
            blended=True,
            overlap_gain=0.4,
            subtitle_type="dialogue",
        )
        assert seg.dubbed_text == "क्या बोला तूने!"
        assert seg.emotion == "angry"
        assert seg.emotion_intensity == 0.9
        assert seg.blended is True
        assert seg.overlap_gain == 0.4


# ---------------------------------------------------------------------------
# Segment default values
# ---------------------------------------------------------------------------

class TestSegmentDefaults:
    def _make(self):
        return Segment(
            segment_id="seg_000",
            start=0.0,
            end=2.0,
            speaker_id="SPEAKER_00",
            source_text="test",
        )

    def test_translated_text_default(self):
        assert self._make().translated_text == ""

    def test_dubbed_text_default(self):
        assert self._make().dubbed_text == ""

    def test_emotion_default(self):
        assert self._make().emotion == "neutral"

    def test_emotion_intensity_default(self):
        assert self._make().emotion_intensity == 0.0

    def test_voice_profile_default(self):
        assert self._make().voice_profile == ""

    def test_tts_audio_path_default(self):
        assert self._make().tts_audio_path == ""

    def test_stretch_ratio_default(self):
        assert self._make().stretch_ratio == 1.0

    def test_discarded_default(self):
        assert self._make().discarded is False

    def test_blended_default(self):
        assert self._make().blended is False

    def test_overlap_gain_default(self):
        assert self._make().overlap_gain == 1.0

    def test_subtitle_type_default(self):
        assert self._make().subtitle_type == "dialogue"


# ---------------------------------------------------------------------------
# Segment field types
# ---------------------------------------------------------------------------

class TestSegmentFieldTypes:
    def _make(self, **kwargs):
        base = dict(
            segment_id="seg_t",
            start=0.5,
            end=1.5,
            speaker_id="SPEAKER_00",
            source_text="hello",
        )
        base.update(kwargs)
        return Segment(**base)

    def test_segment_id_is_str(self):
        assert isinstance(self._make().segment_id, str)

    def test_start_is_float(self):
        assert isinstance(self._make().start, float)

    def test_end_is_float(self):
        assert isinstance(self._make().end, float)

    def test_speaker_id_is_str(self):
        assert isinstance(self._make().speaker_id, str)

    def test_source_text_is_str(self):
        assert isinstance(self._make().source_text, str)

    def test_emotion_intensity_is_float(self):
        assert isinstance(self._make().emotion_intensity, float)

    def test_stretch_ratio_is_float(self):
        assert isinstance(self._make().stretch_ratio, float)

    def test_discarded_is_bool(self):
        assert isinstance(self._make().discarded, bool)

    def test_blended_is_bool(self):
        assert isinstance(self._make().blended, bool)

    def test_overlap_gain_is_float(self):
        assert isinstance(self._make().overlap_gain, float)

    def test_subtitle_type_is_str(self):
        assert isinstance(self._make().subtitle_type, str)

    def test_start_accepts_int_coercion(self):
        # Python dataclasses don't coerce, but int is a valid float-compatible value
        seg = self._make(start=2, end=4)
        assert seg.start == 2
        assert seg.end == 4


# ---------------------------------------------------------------------------
# Segment missing required fields
# ---------------------------------------------------------------------------

class TestSegmentRequiredFields:
    def test_missing_segment_id_raises(self):
        with pytest.raises(TypeError):
            Segment(start=0.0, end=1.0, speaker_id="SPEAKER_00", source_text="x")  # type: ignore

    def test_missing_start_raises(self):
        with pytest.raises(TypeError):
            Segment(segment_id="s", end=1.0, speaker_id="SPEAKER_00", source_text="x")  # type: ignore

    def test_missing_source_text_raises(self):
        with pytest.raises(TypeError):
            Segment(segment_id="s", start=0.0, end=1.0, speaker_id="SPEAKER_00")  # type: ignore


# ---------------------------------------------------------------------------
# VoiceProfile construction and defaults
# ---------------------------------------------------------------------------

class TestVoiceProfile:
    def _make(self, **kwargs):
        base = dict(
            profile_id="voice_male_deep",
            display_name="Male Deep",
            tts_backend="coqui",
            model_path="tts_models/hi/cv/vits",
            speaker_id="spk_0",
        )
        base.update(kwargs)
        return VoiceProfile(**base)

    def test_construction(self):
        vp = self._make()
        assert vp.profile_id == "voice_male_deep"
        assert vp.tts_backend == "coqui"

    def test_pitch_offset_default(self):
        assert self._make().pitch_offset == 0.0

    def test_rate_multiplier_default(self):
        assert self._make().rate_multiplier == 1.0

    def test_pitch_offset_type(self):
        assert isinstance(self._make().pitch_offset, float)

    def test_rate_multiplier_type(self):
        assert isinstance(self._make().rate_multiplier, float)

    def test_explicit_pitch_and_rate(self):
        vp = self._make(pitch_offset=2.0, rate_multiplier=1.15)
        assert vp.pitch_offset == 2.0
        assert vp.rate_multiplier == 1.15


# ---------------------------------------------------------------------------
# VoiceMap load/save helpers
# ---------------------------------------------------------------------------

class TestVoiceMapHelpers:
    def test_load_returns_none_when_missing(self, tmp_path):
        result = load_voice_map("nonexistent_ep", voice_map_dir=str(tmp_path))
        assert result is None

    def test_save_and_load_roundtrip(self, tmp_path):
        vm: VoiceMap = {
            "episode_id": "ep01",
            "tts_backend": "coqui",
            "mappings": {"SPEAKER_00": "voice_male_deep", "SPEAKER_01": "voice_female_bright"},
        }
        save_voice_map(vm, voice_map_dir=str(tmp_path))
        loaded = load_voice_map("ep01", voice_map_dir=str(tmp_path))
        assert loaded is not None
        assert loaded["episode_id"] == "ep01"
        assert loaded["tts_backend"] == "coqui"
        assert loaded["mappings"]["SPEAKER_00"] == "voice_male_deep"

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "nested" / "voice_maps"
        vm: VoiceMap = {
            "episode_id": "ep02",
            "tts_backend": "gtts",
            "mappings": {},
        }
        save_voice_map(vm, voice_map_dir=str(nested))
        loaded = load_voice_map("ep02", voice_map_dir=str(nested))
        assert loaded is not None

    def test_save_returns_absolute_path(self, tmp_path):
        vm: VoiceMap = {
            "episode_id": "ep03",
            "tts_backend": "nemo",
            "mappings": {},
        }
        path = save_voice_map(vm, voice_map_dir=str(tmp_path))
        import os
        assert os.path.isabs(path)
        assert path.endswith("ep03_voice_map.json")
