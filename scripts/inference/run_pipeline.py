"""
Master Pipeline Runner — orchestrates all stages end to end.
Usage: python scripts/inference/run_pipeline.py --input episode.mp4 --lang hi
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path


def run_stage(label: str, script: str, extra_args: list = []):
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*55}")
    print(f"  ▶  {label}")
    print(f"{'='*55}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {label}")
        sys.exit(1)
    print(f"  ✅ Done: {label}")


def get_audio_duration(path: str) -> float:
    import librosa
    duration = librosa.get_duration(path=path)
    return duration


def main():
    parser = argparse.ArgumentParser(description="🎌 Anime Dubbing AI Pipeline")
    parser.add_argument("--input", required=True, help="Input anime video (.mp4/.mkv) or audio (.wav)")
    parser.add_argument("--lang", default="hi", help="Target language ISO code (hi/pa/ta/bn)")
    parser.add_argument("--bgm", default=None, help="Background music file (optional)")
    parser.add_argument("--speaker-wav", default=None, help="Reference voice for cloning")
    parser.add_argument("--skip-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--start-stage", type=int, default=1, help="Resume from stage N")
    parser.add_argument("--model-size", default="large-v3", help="Whisper model size")
    args = parser.parse_args()

    print(f"\n🎌  Anime Dub AI  |  {args.input}  →  [{args.lang.upper()}]")

    for d in ["data/processed", "data/tts_output", "outputs", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.start_stage <= 1:
        run_stage("Stage 1: Audio Extraction + Preprocessing",
                  "scripts/preprocessing/extract_audio.py",
                  ["--input", args.input, "--output", "data/processed/audio.wav"])

    if args.start_stage <= 2 and not args.skip_diarize:
        run_stage("Stage 2: Speaker Diarization",
                  "scripts/preprocessing/diarize_speakers.py",
                  ["--audio", "data/processed/audio.wav"])

    if args.start_stage <= 3:
        run_stage("Stage 3: ASR — Japanese Transcription",
                  "scripts/preprocessing/asr_transcribe.py",
                  ["--audio", "data/processed/audio.wav",
                   "--output", "data/processed/transcript_ja.json",
                   "--model", args.model_size])

    if args.start_stage <= 4:
        run_stage(f"Stage 4: Translation (ja → {args.lang})",
                  "scripts/preprocessing/translate.py",
                  ["--input", "data/processed/transcript_ja.json",
                   "--output", f"data/processed/transcript_{args.lang}.json",
                   "--src", "ja", "--tgt", args.lang])

    if args.start_stage <= 5:
        tts_args = ["--input", f"data/processed/transcript_{args.lang}.json",
                    "--output-dir", "data/tts_output/"]
        if args.speaker_wav:
            tts_args += ["--speaker-wav", args.speaker_wav]
        run_stage("Stage 5: Hindi TTS Synthesis", "scripts/inference/tts_hindi.py", tts_args)

    if args.start_stage <= 6:
        duration = get_audio_duration("data/processed/audio.wav")
        mix_args = ["--segments", "data/tts_output/segments.json",
                    "--output", f"outputs/final_{args.lang}_dub.wav",
                    "--duration", str(duration)]
        if args.bgm:
            mix_args += ["--bgm", args.bgm]
        run_stage("Stage 6: Time Alignment + Final Mix",
                  "scripts/inference/align_and_mix.py", mix_args)

    print(f"\n🎉  Pipeline complete!")
    print(f"    Output → outputs/final_{args.lang}_dub.wav\n")


if __name__ == "__main__":
    main()
