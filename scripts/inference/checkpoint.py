"""
Checkpoint management for the production dubbing pipeline.

Handles writing and reading checkpoint JSON files so that a failed pipeline
run can be resumed from the last successfully completed stage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.inference.exceptions import PipelineResumeError

logger = logging.getLogger(__name__)

# Maps stage numbers 2–10 to their required input file path templates.
# Template variables: {episode_id}
STAGE_INPUTS: Dict[int, List[str]] = {
    # Stage 2 (Source Separation) needs the raw extracted audio from Stage 1.
    2: ["data/raw_audio/extracted.wav"],
    # Stage 3 (Diarization) needs the extracted audio (or vocals stem if separation ran).
    3: ["data/raw_audio/extracted.wav"],
    # Stage 4 (Translation) needs the diarization output from Stage 3.
    4: ["data/processed/diarization.json"],
    # Stage 5 (Emotion Detection) needs a translated transcript.
    # We check for at least one language transcript; hi is the primary target.
    5: ["data/processed/transcript_hi.json"],
    # Stage 6 (Voice Assignment) needs the emotion-annotated transcript.
    6: ["data/processed/transcript_hi.json"],
    # Stage 7 (TTS Synthesis) needs the voice map produced by Stage 6.
    7: ["data/voice_maps/{episode_id}_voice_map.json"],
    # Stage 8 (Mixing) needs the background stem and the processed audio.
    8: [
        "data/processed/background.wav",
        "data/processed/transcript_hi.json",
    ],
    # Stage 9 (Muxing) needs the final mixed Hindi audio from Stage 8.
    9: ["data/processed/final_hindi_dubbed.wav"],
    # Stage 10 (Validation) needs the muxed output video.
    10: ["outputs/{episode_id}_dubbed.mp4"],
}


class CheckpointManager:
    """Manages pipeline checkpoints for failure recovery and resume support.

    Checkpoints are written to ``data/processed/{episode_id}_checkpoint.json``
    after each successfully completed stage.  On the next run the orchestrator
    reads the checkpoint to determine which stage to start from.

    Checkpoint schema::

        {
            "episode_id":        str,
            "completed_stages":  list[int],
            "last_stage":        int,
            "timestamp":         str,   # ISO-8601
            "input_file":        str
        }
    """

    CHECKPOINT_DIR = Path("data/processed")

    def _checkpoint_path(self, episode_id: str) -> Path:
        return self.CHECKPOINT_DIR / f"{episode_id}_checkpoint.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_checkpoint(
        self,
        episode_id: str,
        completed_stages: List[int],
        input_file: str,
    ) -> None:
        """Write a checkpoint file recording completed stages.

        Parameters
        ----------
        episode_id:
            Identifier for the episode (e.g. ``"ep01"``).
        completed_stages:
            Ordered list of stage numbers that have finished successfully.
        input_file:
            Path to the original input file for this episode.
        """
        last_stage = completed_stages[-1] if completed_stages else 0
        checkpoint = {
            "episode_id": episode_id,
            "completed_stages": completed_stages,
            "last_stage": last_stage,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_file": input_file,
        }
        path = self._checkpoint_path(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(
            "[Checkpoint] Written for episode '%s': stages %s → %s",
            episode_id,
            completed_stages,
            path,
        )

    def read_checkpoint(self, episode_id: str) -> Optional[dict]:
        """Read the checkpoint file for *episode_id*.

        Parameters
        ----------
        episode_id:
            Identifier for the episode.

        Returns
        -------
        dict or None
            The parsed checkpoint dict, or ``None`` if no checkpoint exists.
        """
        path = self._checkpoint_path(episode_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[Checkpoint] Failed to decode checkpoint for episode '%s': %s",
                episode_id,
                exc,
            )
            return None

    def determine_start_stage(
        self,
        episode_id: str,
        resume_from: Optional[int] = None,
        force_restart: bool = False,
    ) -> int:
        """Determine which pipeline stage to start from.

        Logic (in priority order):

        1. ``force_restart=True`` → return 1 (ignore any checkpoint).
        2. ``resume_from`` is given → return ``resume_from``.
        3. Checkpoint exists → return ``last_stage + 1``.
        4. No checkpoint → return 1.

        Parameters
        ----------
        episode_id:
            Identifier for the episode.
        resume_from:
            Explicit stage number to resume from (CLI ``--resume-from``).
        force_restart:
            If ``True``, restart from stage 1 regardless of checkpoint.

        Returns
        -------
        int
            The stage number to begin execution from.
        """
        if force_restart:
            logger.info(
                "[Checkpoint] --force-restart: starting from stage 1 for episode '%s'",
                episode_id,
            )
            return 1

        if resume_from is not None:
            logger.info(
                "[Checkpoint] --resume-from %d: starting from stage %d for episode '%s'",
                resume_from,
                resume_from,
                episode_id,
            )
            return resume_from

        checkpoint = self.read_checkpoint(episode_id)
        if checkpoint is not None:
            last_stage = checkpoint.get("last_stage", 0)
            start_stage = last_stage + 1
            logger.info(
                "[Checkpoint] Resuming episode '%s' from stage %d (last completed: %d)",
                episode_id,
                start_stage,
                last_stage,
            )
            return start_stage

        logger.info(
            "[Checkpoint] No checkpoint found for episode '%s': starting from stage 1",
            episode_id,
        )
        return 1

    def validate_stage_inputs(self, stage: int, episode_id: str) -> None:
        """Verify that all required input files for *stage* exist on disk.

        Uses :data:`STAGE_INPUTS` to look up the expected file paths for the
        given stage, substituting ``{episode_id}`` in each template.

        Parameters
        ----------
        stage:
            Pipeline stage number (2–10).
        episode_id:
            Identifier for the episode.

        Raises
        ------
        PipelineResumeError
            If one or more required input files are absent, with the list of
            missing paths included in the exception.
        """
        templates = STAGE_INPUTS.get(stage, [])
        if not templates:
            logger.debug(
                "[Checkpoint] No input requirements defined for stage %d — skipping validation",
                stage,
            )
            return

        missing: List[str] = []
        for template in templates:
            resolved = template.format(episode_id=episode_id)
            if not Path(resolved).exists():
                missing.append(resolved)

        if missing:
            logger.error(
                "[Checkpoint] Stage %d for episode '%s' is missing %d required file(s): %s",
                stage,
                episode_id,
                len(missing),
                missing,
            )
            raise PipelineResumeError(missing)

        logger.debug(
            "[Checkpoint] Stage %d input validation passed for episode '%s'",
            stage,
            episode_id,
        )
