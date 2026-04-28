"""
Pipeline cache layer for the production dubbing pipeline.

Manages SHA256-keyed cache for translations, embeddings, and TTS audio
in data/cache/ subdirectories:
  - data/cache/translations/  — JSON files keyed by SHA256 of (src_text, src_lang, tgt_lang, backend)
  - data/cache/embeddings/    — .npy files keyed by SHA256 of (audio_path, mtime)
  - data/cache/tts/           — .wav files keyed by SHA256 of (dubbed_text, voice_profile_id,
                                  emotion, emotion_intensity, tts_backend)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PipelineCache:
    """SHA256-keyed cache for translations, speaker embeddings, and TTS audio.

    Cache files are stored under three subdirectories of ``data/cache/``:

    - ``translations/`` — JSON blobs
    - ``embeddings/``   — NumPy ``.npy`` arrays
    - ``tts/``          — WAV audio files

    All directories are created on instantiation if they do not already exist.
    """

    def __init__(self, cache_root: str = "data/cache") -> None:
        self.translations_dir = os.path.join(cache_root, "translations")
        self.embeddings_dir = os.path.join(cache_root, "embeddings")
        self.tts_dir = os.path.join(cache_root, "tts")

        os.makedirs(self.translations_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.tts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def cache_key(self, *parts: str) -> str:
        """Return a hex SHA256 digest of the concatenated *parts*.

        Parameters
        ----------
        *parts:
            Arbitrary string components that together uniquely identify
            a cache entry (e.g. source text, language codes, backend name).

        Returns
        -------
        str
            64-character lowercase hex string.
        """
        combined = "\x00".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Translation cache
    # ------------------------------------------------------------------

    def get_translation(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        backend: str,
    ) -> dict | None:
        """Look up a cached translation.

        Parameters
        ----------
        src_text:
            Source language text to translate.
        src_lang:
            BCP-47 source language code (e.g. ``"ja"``).
        tgt_lang:
            BCP-47 target language code (e.g. ``"hi"``).
        backend:
            Translation backend identifier (e.g. ``"Helsinki-NLP/opus-mt-ja-hi"``).

        Returns
        -------
        dict or None
            Cached translation payload, or ``None`` on a cache miss.
            Logs ``[CACHE HIT] translation {key[:8]}`` on a hit.
        """
        key = self.cache_key(src_text, src_lang, tgt_lang, backend)
        cache_path = os.path.join(self.translations_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("[CACHE HIT] translation %s", key[:8])
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read translation cache entry %s: %s", key[:8], exc)
            return None

    def set_translation(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        backend: str,
        data: dict,
    ) -> None:
        """Persist a translation result to the cache.

        Parameters
        ----------
        src_text:
            Source language text that was translated.
        src_lang:
            BCP-47 source language code.
        tgt_lang:
            BCP-47 target language code.
        backend:
            Translation backend identifier.
        data:
            Translation payload to cache (must be JSON-serialisable).
        """
        key = self.cache_key(src_text, src_lang, tgt_lang, backend)
        cache_path = os.path.join(self.translations_dir, f"{key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("Cached translation %s → %s", key[:8], cache_path)
        except OSError as exc:
            logger.warning("Failed to write translation cache entry %s: %s", key[:8], exc)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_embedding(self, audio_path: str, mtime: float) -> np.ndarray | None:
        """Look up a cached speaker embedding.

        Parameters
        ----------
        audio_path:
            Path to the audio file whose embedding was computed.
        mtime:
            File modification time (``os.path.getmtime``) used to
            invalidate stale cache entries.

        Returns
        -------
        np.ndarray or None
            Cached embedding array, or ``None`` on a cache miss.
            Logs ``[CACHE HIT] embedding {key[:8]}`` on a hit.
        """
        key = self.cache_key(audio_path, str(mtime))
        cache_path = os.path.join(self.embeddings_dir, f"{key}.npy")
        if not os.path.exists(cache_path):
            return None
        try:
            embedding = np.load(cache_path)
            logger.info("[CACHE HIT] embedding %s", key[:8])
            return embedding
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read embedding cache entry %s: %s", key[:8], exc)
            return None

    def set_embedding(
        self,
        audio_path: str,
        mtime: float,
        embedding: np.ndarray,
    ) -> None:
        """Persist a speaker embedding to the cache.

        Parameters
        ----------
        audio_path:
            Path to the audio file whose embedding was computed.
        mtime:
            File modification time used as part of the cache key.
        embedding:
            NumPy array to store as a ``.npy`` file.
        """
        key = self.cache_key(audio_path, str(mtime))
        cache_path = os.path.join(self.embeddings_dir, f"{key}.npy")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        try:
            np.save(cache_path, embedding)
            logger.info("[CACHE SET] embedding %s", key[:8])
        except OSError as exc:
            logger.warning("Failed to write embedding cache entry %s: %s", key[:8], exc)

    # ------------------------------------------------------------------
    # TTS cache
    # ------------------------------------------------------------------

    def get_tts(
        self,
        dubbed_text: str,
        voice_profile_id: str,
        emotion: str,
        emotion_intensity: float,
        tts_backend: str,
    ) -> str | None:
        """Look up a cached TTS WAV file path.

        Parameters
        ----------
        dubbed_text:
            Hindi dubbed text that was synthesised.
        voice_profile_id:
            Voice profile identifier (from ``character_voices.yaml``).
        emotion:
            Emotion label (e.g. ``"angry"``, ``"neutral"``).
        emotion_intensity:
            Emotion intensity in ``[0.0, 1.0]``.
        tts_backend:
            TTS backend identifier (e.g. ``"coqui"``, ``"gtts"``).

        Returns
        -------
        str or None
            Path to the cached WAV file, or ``None`` on a cache miss.
            Logs ``[CACHE HIT] tts {key[:8]}`` on a hit.
        """
        key = self.cache_key(
            dubbed_text,
            voice_profile_id,
            emotion,
            str(emotion_intensity),
            tts_backend,
        )
        cache_path = os.path.join(self.tts_dir, f"{key}.wav")
        if not os.path.exists(cache_path):
            return None
        logger.info("[CACHE HIT] tts %s", key[:8])
        return cache_path

    def set_tts(
        self,
        dubbed_text: str,
        voice_profile_id: str,
        emotion: str,
        emotion_intensity: float,
        tts_backend: str,
        wav_path: str,
    ) -> str:
        """Copy a synthesised WAV into the TTS cache.

        Parameters
        ----------
        dubbed_text:
            Hindi dubbed text that was synthesised.
        voice_profile_id:
            Voice profile identifier.
        emotion:
            Emotion label.
        emotion_intensity:
            Emotion intensity in ``[0.0, 1.0]``.
        tts_backend:
            TTS backend identifier.
        wav_path:
            Path to the source WAV file to copy into the cache.

        Returns
        -------
        str
            Path to the cached WAV file.
        """
        key = self.cache_key(
            dubbed_text,
            voice_profile_id,
            emotion,
            str(emotion_intensity),
            tts_backend,
        )
        cache_path = os.path.join(self.tts_dir, f"{key}.wav")
        os.makedirs(self.tts_dir, exist_ok=True)
        shutil.copy2(wav_path, cache_path)
        logger.info("[CACHE SET] tts %s", key[:8])
        return cache_path

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> int:
        """Delete all cached files across all three subdirectories.

        Returns
        -------
        int
            Total number of files deleted.
        """
        count = 0
        for subdir in (self.translations_dir, self.embeddings_dir, self.tts_dir):
            for entry in os.scandir(subdir):
                if entry.is_file():
                    os.remove(entry.path)
                    count += 1
        logger.info("[CACHE] Cleared %d cached file(s).", count)
        return count
