# Spec: Phase 7 — Testing & CI

## Overview
Add a `tests/` folder with unit tests for each pipeline stage and an automated quality benchmark that tracks PESQ/STOI scores over time. Phase 1 must be complete before starting this phase.

**Prerequisite:** A 30-second sample clip processed through Phase 1 so test fixtures exist.

---

## Requirements

1. Unit tests cover all 6 pipeline stages plus subtitle generation
2. All tests use a shared 30-second fixture clip — not a full episode
3. Tests assert on file properties (sample rate, duration, JSON keys) not on content
4. `pytest tests/ -v` passes in under 5 minutes on CPU
5. A benchmark script logs PESQ and STOI to timestamped files in `logs/`

---

## Tasks

### Task 1: Set Up Test Infrastructure
Create `tests/conftest.py` with shared fixtures.

```python
import pytest
import subprocess, sys
from pathlib import Path

SAMPLE_CLIP = "tests/fixtures/sample_30s.mp4"   # 30-second test clip
PROCESSED_DIR = "tests/fixtures/processed/"

@pytest.fixture(scope="session", autouse=True)
def run_stage1():
    """Extract audio once for the whole test session."""
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "scripts/preprocessing/extract_audio.py",
        "--input", SAMPLE_CLIP,
        "--output", f"{PROCESSED_DIR}/audio.wav"
    ], check=True)
```

Also add a `tests/fixtures/` folder — place a short 30-second anime clip there as `sample_30s.mp4`. This file should be committed to the repo (keep it small — ≤5 MB).

**Done when:** `pytest tests/conftest.py` runs without error.

---

### Task 2: Test Audio Extraction
Create `tests/test_extract_audio.py`.

```python
import librosa
from pathlib import Path

AUDIO_OUT = "tests/fixtures/processed/audio.wav"

def test_output_exists():
    assert Path(AUDIO_OUT).exists()

def test_sample_rate():
    _, sr = librosa.load(AUDIO_OUT, sr=None)
    assert sr == 22050

def test_mono():
    audio, _ = librosa.load(AUDIO_OUT, mono=False)
    # mono = 1D array
    assert audio.ndim == 1

def test_non_empty():
    audio, sr = librosa.load(AUDIO_OUT)
    assert len(audio) / sr > 1.0  # at least 1 second

def test_duration_reasonable():
    duration = librosa.get_duration(path=AUDIO_OUT)
    assert 20 < duration < 40  # 30s fixture ± 10s tolerance
```

**Done when:** All 5 assertions pass.

---

### Task 3: Test ASR Output
Create `tests/test_asr.py`.

Run ASR on the fixture audio in a session-scoped fixture, then assert on the output JSON.

```python
import json, subprocess, sys
import pytest

ASR_OUT = "tests/fixtures/processed/transcript_ja.json"

@pytest.fixture(scope="module")
def transcript():
    subprocess.run([
        sys.executable, "scripts/preprocessing/asr_transcribe.py",
        "--audio", "tests/fixtures/processed/audio.wav",
        "--output", ASR_OUT,
        "--model", "medium"
    ], check=True)
    with open(ASR_OUT, encoding="utf-8") as f:
        return json.load(f)

def test_has_segments(transcript):
    assert len(transcript) > 0

def test_segment_keys(transcript):
    for seg in transcript:
        assert "start" in seg and "end" in seg and "text" in seg

def test_timestamps_ordered(transcript):
    starts = [s["start"] for s in transcript]
    assert starts == sorted(starts)

def test_has_japanese_text(transcript):
    # Check at least one segment contains a Japanese character
    all_text = " ".join(s["text"] for s in transcript)
    assert any('\u3000' <= c <= '\u9fff' for c in all_text)
```

**Done when:** All 4 assertions pass.

---

### Task 4: Test Translation Output
Create `tests/test_translate.py`.

```python
import json, subprocess, sys
import pytest

TRANS_OUT = "tests/fixtures/processed/transcript_hi.json"

@pytest.fixture(scope="module")
def translated(transcript_ja):
    subprocess.run([
        sys.executable, "scripts/preprocessing/translate.py",
        "--input", "tests/fixtures/processed/transcript_ja.json",
        "--output", TRANS_OUT,
    ], check=True)
    with open(TRANS_OUT, encoding="utf-8") as f:
        return json.load(f)

def test_no_segments_dropped(translated, transcript_ja):
    # Allow up to 5% drop due to empty/error segments
    assert len(translated) >= len(transcript_ja) * 0.95

def test_translation_fields(translated):
    for seg in translated:
        assert "text_translated" in seg

def test_cleaned_field(translated):
    for seg in translated:
        assert "text_cleaned" in seg  # from Task 1.8

def test_no_empty_translations(translated):
    empty = [s for s in translated if not s.get("text_translated", "").strip()]
    assert len(empty) < len(translated) * 0.10  # <10% empty
```

**Done when:** All 4 assertions pass.

---

### Task 5: Test TTS Output
Create `tests/test_tts.py`.

```python
import json
from pathlib import Path
import librosa

SEGMENTS_JSON = "tests/fixtures/tts_output/segments.json"

def test_segments_manifest_exists():
    assert Path(SEGMENTS_JSON).exists()

def test_all_audio_files_exist():
    with open(SEGMENTS_JSON) as f:
        segs = json.load(f)
    missing = [s["audio_file"] for s in segs if not Path(s["audio_file"]).exists()]
    assert len(missing) == 0, f"Missing: {missing[:5]}"

def test_audio_files_non_empty():
    with open(SEGMENTS_JSON) as f:
        segs = json.load(f)
    for seg in segs[:5]:  # check first 5
        dur = librosa.get_duration(path=seg["audio_file"])
        assert dur > 0.1

def test_stretch_ratio_field():
    with open(SEGMENTS_JSON) as f:
        segs = json.load(f)
    for seg in segs:
        assert "stretch_ratio" in seg  # from Task 1.10
```

**Done when:** All 4 assertions pass.

---

### Task 6: Test Final Mix
Create `tests/test_align.py`.

```python
import librosa
from pathlib import Path

SOURCE_AUDIO = "tests/fixtures/processed/audio.wav"
FINAL_OUTPUT = "tests/fixtures/outputs/final_hi_dub.wav"

def test_output_exists():
    assert Path(FINAL_OUTPUT).exists()

def test_duration_within_tolerance():
    source_dur = librosa.get_duration(path=SOURCE_AUDIO)
    output_dur = librosa.get_duration(path=FINAL_OUTPUT)
    assert abs(output_dur - source_dur) / source_dur < 0.05  # within 5%

def test_no_clipping():
    import numpy as np
    audio, _ = librosa.load(FINAL_OUTPUT)
    clipped = np.sum(np.abs(audio) > 0.99)
    assert clipped < len(audio) * 0.001  # <0.1% samples clipped
```

**Done when:** All 3 assertions pass.

---

### Task 7: Test Subtitle Generation
Create `tests/test_subtitles.py`.

```python
import re
from pathlib import Path

SRT_FILE = "tests/fixtures/outputs/subtitles_hi.srt"

def test_srt_exists():
    assert Path(SRT_FILE).exists()

def test_srt_has_content():
    content = Path(SRT_FILE).read_text(encoding="utf-8")
    assert len(content.strip()) > 0

def test_srt_format():
    content = Path(SRT_FILE).read_text(encoding="utf-8")
    blocks = content.strip().split("\n\n")
    for block in blocks[:5]:  # check first 5 blocks
        lines = block.strip().splitlines()
        assert lines[0].isdigit(), f"Expected index, got: {lines[0]}"
        timestamp_pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        assert re.match(timestamp_pattern, lines[1]), f"Bad timestamp: {lines[1]}"
        assert len(lines) >= 3, "Block has no text line"
```

**Done when:** All 3 assertions pass.

---

### Task 8: Create Quality Benchmark Script
Create `scripts/evaluation/benchmark.py`.

```python
from scripts.evaluation.evaluate_quality import evaluate
from datetime import datetime
from pathlib import Path
import json, argparse

def run_benchmark(test_pairs: list, output_log: str):
    """
    test_pairs = [{"original": "path/to/orig.wav", "dubbed": "path/to/dubbed.wav"}]
    """
    results = []
    for pair in test_pairs:
        metrics = evaluate(pair["original"], pair["dubbed"])
        metrics["pair"] = pair
        results.append(metrics)

    avg_pesq = sum(r["pesq_score"] for r in results) / len(results)
    avg_stoi = sum(r["stoi_score"] for r in results) / len(results)

    log = {
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(results),
        "avg_pesq": round(avg_pesq, 4),
        "avg_stoi": round(avg_stoi, 4),
        "results": results,
    }

    Path(output_log).parent.mkdir(parents=True, exist_ok=True)
    with open(output_log, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n── Benchmark Results ──────────────────")
    print(f"  Pairs evaluated : {len(results)}")
    print(f"  Avg PESQ        : {avg_pesq:.4f}  (range: 1–4.5, higher=better)")
    print(f"  Avg STOI        : {avg_stoi:.4f}  (range: 0–1, higher=better)")
    print(f"  Log saved       : {output_log}")
```

Run:
```bash
python scripts/evaluation/benchmark.py \
  --pairs tests/fixtures/benchmark_pairs.json \
  --output "logs/benchmark_$(date +%F).json"
```

Where `benchmark_pairs.json` is a list of `{"original": "...", "dubbed": "..."}` pairs.

**Done when:** Script runs, prints the table, saves timestamped JSON to `logs/`.

---

## Acceptance Criteria

- [ ] `tests/fixtures/sample_30s.mp4` — committed to repo, ≤5 MB
- [ ] `pytest tests/ -v` — all tests pass in under 5 minutes on CPU
- [ ] `test_extract_audio.py` — 5 tests pass
- [ ] `test_asr.py` — 4 tests pass (Japanese characters detected)
- [ ] `test_translate.py` — 4 tests pass (`text_cleaned` field present)
- [ ] `test_tts.py` — 4 tests pass (`stretch_ratio` field present)
- [ ] `test_align.py` — 3 tests pass (duration within 5%, clipping <0.1%)
- [ ] `test_subtitles.py` — 3 tests pass (valid SRT format)
- [ ] `scripts/evaluation/benchmark.py` — runs, prints table, saves to `logs/`
