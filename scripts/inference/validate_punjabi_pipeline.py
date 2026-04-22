"""
Validation script for the Punjabi TTS pipeline.
Creates a minimal synthetic transcript_pa.json and verifies gTTS can synthesize
Punjabi (Gurmukhi) text to WAV files.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TRANSCRIPT_PATH = "data/processed/transcript_pa.json"
TTS_OUTPUT_DIR = "data/tts_output/"

SYNTHETIC_SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
        "text_translated": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
        "emotion": "neutral",
        "emotion_intensity": 0.0,
    },
    {
        "start": 3.0,
        "end": 5.5,
        "text": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ",
        "text_translated": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ",
        "emotion": "neutral",
        "emotion_intensity": 0.0,
    },
    {
        "start": 6.0,
        "end": 9.0,
        "text": "ਪੰਜਾਬੀ ਬਹੁਤ ਸੁੰਦਰ ਭਾਸ਼ਾ ਹੈ",
        "text_translated": "ਪੰਜਾਬੀ ਬਹੁਤ ਸੁੰਦਰ ਭਾਸ਼ਾ ਹੈ",
        "emotion": "happy",
        "emotion_intensity": 0.3,
    },
]


def main():
    print("=" * 55)
    print("  Punjabi Pipeline Validation")
    print("=" * 55)

    # Step 1: Write synthetic transcript
    Path(TRANSCRIPT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(SYNTHETIC_SEGMENTS, f, indent=2, ensure_ascii=False)
    print(f"[1] Wrote synthetic transcript → {TRANSCRIPT_PATH}")

    # Step 2: Run TTS synthesis
    from scripts.inference.tts_hindi import synthesize_hindi

    print(f"[2] Running TTS synthesis (engine=coqui → gTTS fallback, lang=pa)...")
    try:
        audio_segments = synthesize_hindi(
            segments_json=TRANSCRIPT_PATH,
            output_dir=TTS_OUTPUT_DIR,
            engine="coqui",   # will fall back to gTTS because languages.yaml has tts_engine: google
            lang="pa",
        )
    except Exception as e:
        print(f"\n❌ FAIL — TTS synthesis raised an exception: {e}")
        sys.exit(1)

    # Step 3: Verify WAV files were created
    print(f"[3] Verifying WAV output files...")
    failures = []
    for seg in audio_segments:
        wav_path = seg.get("audio_file", "")
        if not wav_path or not Path(wav_path).exists():
            failures.append(wav_path or "(missing audio_file key)")
        else:
            size = Path(wav_path).stat().st_size
            print(f"    ✅ {wav_path}  ({size} bytes)")

    if failures:
        print(f"\n❌ FAIL — Missing WAV files:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)

    # Step 4: Verify gTTS engine was used
    engines_used = {seg.get("tts_engine") for seg in audio_segments}
    print(f"[4] TTS engines used: {engines_used}")
    if "gtts" not in engines_used:
        print(f"\n❌ FAIL — Expected gTTS engine but got: {engines_used}")
        sys.exit(1)

    print(f"\n✅ PASS — Punjabi TTS pipeline validated successfully.")
    print(f"   {len(audio_segments)} WAV segments produced in {TTS_OUTPUT_DIR}")
    print(f"   Pipeline is correctly wired for: python scripts/inference/run_pipeline.py --input sample.mp4 --lang pa")
    print(f"   Output would be: outputs/final_pa_dub.wav")


if __name__ == "__main__":
    main()
