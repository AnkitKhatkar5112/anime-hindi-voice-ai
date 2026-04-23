import pytest
import subprocess
import sys
from pathlib import Path

SAMPLE_CLIP = "tests/fixtures/sample_30s.mp4"
PROCESSED_DIR = "tests/fixtures/processed"

@pytest.fixture(scope="session", autouse=True)
def run_stage1():
    """Extract audio once for the whole test session (skips if already exists)."""
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    audio_out = Path(f"{PROCESSED_DIR}/audio.wav")
    if audio_out.exists() and audio_out.stat().st_size > 0:
        return  # Already extracted, skip re-running
    subprocess.run([
        sys.executable, "scripts/preprocessing/extract_audio.py",
        "--input", SAMPLE_CLIP,
        "--output", str(audio_out)
    ], check=True)
