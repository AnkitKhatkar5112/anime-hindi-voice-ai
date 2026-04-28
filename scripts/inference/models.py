"""
Core data models for the production dubbing pipeline.

This module defines the shared dataclasses used across all pipeline stages,
from ingest through muxing and subtitle generation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict


@dataclass
class Segment:
    """Represents a single timed speech segment throughout the pipeline."""

    # --- Required fields (no defaults) ---
    segment_id: str           # e.g. "seg_001"
    start: float              # seconds
    end: float                # seconds
    speaker_id: str           # e.g. "SPEAKER_00"
    source_text: str          # original Japanese / source language text

    # --- Fields with defaults ---
    translated_text: str = ""         # literal translation
    dubbed_text: str = ""             # localized Hindi dubbing text
    emotion: str = "neutral"          # neutral | happy | angry | sad | excited | fearful
    emotion_intensity: float = 0.0    # 0.0 – 1.0
    voice_profile: str = ""           # key from character_voices.yaml
    tts_audio_path: str = ""          # path to synthesized WAV
    stretch_ratio: float = 1.0        # 1.0 = no stretch
    discarded: bool = False           # True if removed by overlap resolution
    blended: bool = False             # True if volume-reduced during overlap blend
    overlap_gain: float = 1.0         # gain applied during overlap window (default 1.0)
    subtitle_type: str = "dialogue"   # "dialogue" | "sign" | "mixed" (set by Subtitle_Classifier)


@dataclass
class VoiceProfile:
    """Represents a TTS voice profile assigned to a speaker."""

    profile_id: str           # e.g. "voice_male_deep"
    display_name: str
    tts_backend: str          # "nemo" | "coqui" | "gtts" | "bark"
    model_path: str           # path or model name
    speaker_id: str           # speaker embedding ID within model
    pitch_offset: float = 0.0    # semitones, relative to model default
    rate_multiplier: float = 1.0  # 1.0 = default


class VoiceMap(TypedDict, total=False):
    """
    Schema for the per-episode voice map persisted to
    ``data/voice_maps/{episode_id}_voice_map.json``.

    Required keys
    -------------
    episode_id : str
        Identifier for the episode (e.g. ``"ep01"``).
    tts_backend : str
        TTS backend locked for this episode: ``"nemo"``, ``"coqui"``,
        ``"gtts"``, or ``"bark"``.
    mappings : Dict[str, str]
        Maps ``SPEAKER_XX`` IDs to voice profile keys from
        ``configs/character_voices.yaml``.

    Optional keys
    -------------
    series_id : str
        Series identifier used when sharing a voice map across episodes.
    created_at : str
        ISO-8601 timestamp of when the map was first created.
    """

    episode_id: str
    tts_backend: str
    mappings: Dict[str, str]
    series_id: str
    created_at: str


_VOICE_MAP_DIR = "data/voice_maps"


def load_voice_map(episode_id: str, voice_map_dir: str = _VOICE_MAP_DIR) -> Optional[VoiceMap]:
    """Load the voice map for *episode_id* from disk.

    Parameters
    ----------
    episode_id:
        Episode identifier (e.g. ``"ep01"``).
    voice_map_dir:
        Directory that contains voice map JSON files.
        Defaults to ``data/voice_maps``.

    Returns
    -------
    VoiceMap or None
        The parsed voice map dict, or ``None`` if no file exists yet.
    """
    path = os.path.join(voice_map_dir, f"{episode_id}_voice_map.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_voice_map(voice_map: VoiceMap, voice_map_dir: str = _VOICE_MAP_DIR) -> str:
    """Persist *voice_map* to disk.

    Parameters
    ----------
    voice_map:
        The voice map to save.  Must contain at least ``episode_id``,
        ``tts_backend``, and ``mappings`` keys.
    voice_map_dir:
        Directory where the JSON file will be written.
        Defaults to ``data/voice_maps``.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    os.makedirs(voice_map_dir, exist_ok=True)
    episode_id = voice_map["episode_id"]
    path = os.path.join(voice_map_dir, f"{episode_id}_voice_map.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(voice_map, fh, indent=2, ensure_ascii=False)
    return os.path.abspath(path)
