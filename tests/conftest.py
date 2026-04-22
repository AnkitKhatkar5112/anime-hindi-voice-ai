import pytest
import subprocess
import sys
from pathlib import Path

SAMPLE_CLIP = "tests/fixtures/sample_30s.mp4"
PROCESSED_DIR = "tests/fixtures/processed"

@pytest.fixture(scope="session", autouse=True)
def run_stage1():
    """Extract audio once for the whole test session."""
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "scripts/preprocessing/extract_audio.py",
        "--input", SAMPLE_CLIP,
        "--output", f"{PROCESSED_DIR}/audio.wav"
    ], check=True)
