"""
scripts/training/prepare_dataset.py
Prepare audio dataset in LJ-Speech format for TTS fine-tuning.

Data sources:
  - IITM Hindi TTS: https://www.iitm.ac.in/donlab/tts/
  - AIR corpus: All India Radio broadcast recordings
  - Custom recordings: Studio-recorded voice actor samples (preferred for character voices)

Output format (LJ-Speech):
  output-dir/
    wavs/
      {id}.wav          # 22050 Hz mono, silence-trimmed
    metadata.csv        # pipe-delimited: id|text|text
"""

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ── constants ────────────────────────────────────────────────────────────────
TARGET_SR = 22050
MIN_DURATION_S = 1.0
MAX_DURATION_S = 15.0
SILENCE_TOP_DB = 30
RMS_THRESHOLD = 0.01          # files below this RMS are considered too quiet
WARN_MIN_HOURS = 2.0


# ── transcript loading ────────────────────────────────────────────────────────

def load_transcripts(path: str) -> dict:
    """Return {stem: text} from a CSV or JSON transcript file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcripts file not found: {path}")

    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        # accept {filename: text} or [{filename: ..., text: ...}]
        if isinstance(data, list):
            return {Path(item["filename"]).stem: item["text"] for item in data}
        return {Path(k).stem: v for k, v in data.items()}

    # CSV — try to auto-detect columns
    with open(p, encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t")
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError("Transcript CSV is empty.")

    # find filename and text columns (case-insensitive)
    headers = {h.lower(): h for h in rows[0].keys()}
    fn_col = next((headers[k] for k in ("filename", "file", "id", "name") if k in headers), None)
    tx_col = next((headers[k] for k in ("text", "transcript", "transcription", "normalized_text") if k in headers), None)

    if fn_col is None or tx_col is None:
        raise ValueError(
            f"Cannot detect filename/text columns in CSV. Found: {list(rows[0].keys())}"
        )

    return {Path(row[fn_col]).stem: row[tx_col] for row in rows}


# ── audio processing ──────────────────────────────────────────────────────────

def process_file(audio_path: Path, transcripts: dict, out_wavs: Path):
    """
    Process a single audio file.

    Returns:
        (id, text, duration_s)  on success
        (None, reason, None)    on skip
    """
    stem = audio_path.stem

    text = transcripts.get(stem)
    if text is None:
        return None, f"no transcript for '{stem}'", None

    # load + resample to mono 22050 Hz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

    # trim silence
    audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)

    duration = len(audio) / TARGET_SR

    if duration < MIN_DURATION_S:
        return None, f"too short ({duration:.2f}s < {MIN_DURATION_S}s)", None
    if duration > MAX_DURATION_S:
        return None, f"too long ({duration:.2f}s > {MAX_DURATION_S}s)", None

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < RMS_THRESHOLD:
        return None, f"too quiet (RMS {rms:.5f} < {RMS_THRESHOLD})", None

    out_path = out_wavs / f"{stem}.wav"
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

    return stem, text, duration


# ── reporting ─────────────────────────────────────────────────────────────────

def print_histogram(durations: list, bucket_size: float = 1.0):
    """Print a simple ASCII duration histogram (files per 1-second bucket)."""
    if not durations:
        return
    max_dur = math.ceil(max(durations))
    buckets = {}
    for d in durations:
        b = int(d // bucket_size) * bucket_size
        buckets[b] = buckets.get(b, 0) + 1

    print("\nDuration histogram (seconds):")
    for b in sorted(buckets):
        label = f"  {b:4.0f}–{b + bucket_size:<4.0f}s"
        bar = "█" * buckets[b]
        print(f"{label} | {bar} ({buckets[b]})")


def print_summary(processed: list, skipped: list):
    """Print processing summary and flag short datasets."""
    total_duration_s = sum(d for _, _, d in processed)
    total_hours = total_duration_s / 3600

    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"  Files processed : {len(processed)}")
    print(f"  Files skipped   : {len(skipped)}")
    print(f"  Total duration  : {total_hours:.3f} hours ({total_duration_s:.1f}s)")

    if skipped:
        print("\nSkipped files:")
        for path, reason in skipped:
            print(f"  ✗ {Path(path).name}: {reason}")

    if processed:
        durations = [d for _, _, d in processed]
        print_histogram(durations)

    if total_hours < WARN_MIN_HOURS:
        print(
            f"\n⚠  WARNING: Total duration ({total_hours:.2f}h) is below the recommended "
            f"minimum of {WARN_MIN_HOURS}h for fine-tuning. "
            "Collect more data for better results."
        )
    else:
        print(f"\n✓ Dataset meets the minimum duration requirement ({WARN_MIN_HOURS}h).")

    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset in LJ-Speech format for TTS fine-tuning."
    )
    parser.add_argument("--audio-dir", required=True,
                        help="Directory containing .wav / .mp3 audio files")
    parser.add_argument("--transcripts", required=True,
                        help="CSV or JSON file mapping filename → transcript text")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (will contain wavs/ and metadata.csv)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    if not audio_dir.is_dir():
        print(f"ERROR: --audio-dir '{audio_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # collect audio files
    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")
    )
    if not audio_files:
        print(f"ERROR: No audio files found in '{audio_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) in '{audio_dir}'.")

    # load transcripts
    try:
        transcripts = load_transcripts(args.transcripts)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(transcripts)} transcript(s) from '{args.transcripts}'.")

    # prepare output dirs
    out_wavs = output_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    processed = []   # [(id, text, duration_s)]
    skipped = []     # [(path, reason)]

    for audio_path in audio_files:
        result_id, result_text, duration = process_file(audio_path, transcripts, out_wavs)
        if result_id is None:
            skipped.append((str(audio_path), result_text))
        else:
            processed.append((result_id, result_text, duration))

    # write metadata.csv in LJ-Speech pipe-delimited format: id|text|text
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        for file_id, text, _ in processed:
            # LJ-Speech format: id|normalized_text|normalized_text
            line = f"{file_id}|{text}|{text}\n"
            f.write(line)

    print(f"\nWrote {len(processed)} entries to '{metadata_path}'.")

    print_summary(processed, skipped)


if __name__ == "__main__":
    main()
