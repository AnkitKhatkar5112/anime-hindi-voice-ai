"""Unit tests for CheckpointManager.read_checkpoint."""

import json
import pytest
from pathlib import Path

from scripts.inference.checkpoint import CheckpointManager
from scripts.inference.exceptions import PipelineResumeError


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Return a CheckpointManager whose CHECKPOINT_DIR points to tmp_path."""
    mgr = CheckpointManager()
    monkeypatch.setattr(CheckpointManager, "CHECKPOINT_DIR", tmp_path)
    return mgr


def test_read_checkpoint_returns_none_when_no_file(manager):
    result = manager.read_checkpoint("ep99")
    assert result is None


def test_read_checkpoint_returns_dict_after_write(manager):
    manager.write_checkpoint("ep01", [1, 2, 3], "episodes/ep01.wav")
    result = manager.read_checkpoint("ep01")
    assert result is not None
    assert result["episode_id"] == "ep01"
    assert result["completed_stages"] == [1, 2, 3]
    assert result["last_stage"] == 3
    assert result["input_file"] == "episodes/ep01.wav"
    assert "timestamp" in result


def test_read_checkpoint_returns_none_on_invalid_json(manager, tmp_path):
    path = tmp_path / "ep02_checkpoint.json"
    path.write_text("not valid json", encoding="utf-8")
    result = manager.read_checkpoint("ep02")
    assert result is None


def test_read_checkpoint_returns_none_for_empty_completed_stages(manager):
    manager.write_checkpoint("ep03", [], "episodes/ep03.wav")
    result = manager.read_checkpoint("ep03")
    assert result is not None
    assert result["completed_stages"] == []
    assert result["last_stage"] == 0


# ---------------------------------------------------------------------------
# determine_start_stage tests (task 3.7)
# ---------------------------------------------------------------------------

def test_determine_start_stage_no_checkpoint_returns_stage_1(manager):
    """(a) No checkpoint on disk → start from stage 1."""
    result = manager.determine_start_stage("ep_new")
    assert result == 1


def test_determine_start_stage_checkpoint_last_stage_3_returns_stage_4(manager):
    """(b) Checkpoint with last_stage=3 → resume from stage 4."""
    manager.write_checkpoint("ep01", [1, 2, 3], "episodes/ep01.wav")
    result = manager.determine_start_stage("ep01")
    assert result == 4


def test_determine_start_stage_resume_from_overrides_checkpoint(manager):
    """(c) --resume-from 5 → start from stage 5, even if checkpoint says otherwise."""
    manager.write_checkpoint("ep01", [1, 2, 3], "episodes/ep01.wav")
    result = manager.determine_start_stage("ep01", resume_from=5)
    assert result == 5


def test_determine_start_stage_force_restart_returns_stage_1(manager):
    """(d) --force-restart → always start from stage 1, ignoring any checkpoint."""
    manager.write_checkpoint("ep01", [1, 2, 3, 4, 5], "episodes/ep01.wav")
    result = manager.determine_start_stage("ep01", force_restart=True)
    assert result == 1


# ---------------------------------------------------------------------------
# validate_stage_inputs tests (task 3.8)
# ---------------------------------------------------------------------------

def test_validate_stage_inputs_missing_file_raises_pipeline_resume_error(manager, monkeypatch):
    """Missing required input file raises PipelineResumeError with descriptive message."""
    # Patch Path.exists to always return False so no files appear present
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(PipelineResumeError) as exc_info:
        manager.validate_stage_inputs(4, "ep01")

    error = exc_info.value
    assert len(error.missing_files) >= 1
    assert any("diarization.json" in f for f in error.missing_files)
    assert "missing" in str(error).lower()
    assert "diarization.json" in str(error)


def test_validate_stage_inputs_missing_file_stage2_raises_pipeline_resume_error(manager, monkeypatch):
    """Stage 2 missing extracted.wav raises PipelineResumeError mentioning the file."""
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(PipelineResumeError) as exc_info:
        manager.validate_stage_inputs(2, "ep01")

    error = exc_info.value
    assert any("extracted.wav" in f for f in error.missing_files)
    assert "extracted.wav" in str(error)


def test_validate_stage_inputs_template_substitution_in_error(manager):
    """Stage 7 uses {episode_id} template; missing file message contains the resolved path."""
    with pytest.raises(PipelineResumeError) as exc_info:
        manager.validate_stage_inputs(7, "ep42")

    error = exc_info.value
    assert any("ep42_voice_map.json" in f for f in error.missing_files)
    assert "ep42_voice_map.json" in str(error)
