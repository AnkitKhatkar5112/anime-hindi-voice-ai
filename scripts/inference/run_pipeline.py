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
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/current_stage.txt", "w", encoding="utf-8") as f:
        f.write(label)

    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*55}")
    print(f"  ▶  {label}")
    print(f"{'='*55}")

    log_path = "logs/pipeline_stdout.txt"
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'='*55}\n  ▶  {label}\n{'='*55}\n")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_file.write(result.stdout or "")

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
    parser.add_argument("--video-output", action="store_true", help="Enable Stage 7: Lip Sync video output")
    parser.add_argument("--face-video", default=None, help="Face video for lip sync (required with --video-output)")
    args = parser.parse_args()

    print(f"\n🎌  Anime Dub AI  |  {args.input}  →  [{args.lang.upper()}]")

    for d in ["data/processed", "data/tts_output", "outputs", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Clear stage tracking files
    Path("logs/current_stage.txt").write_text("Starting...", encoding="utf-8")
    Path("logs/pipeline_stdout.txt").write_text("", encoding="utf-8")

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

    # Stage 4b: Dialect post-processing (only when configured in languages.yaml)
    if args.start_stage <= 4:
        import yaml
        with open("configs/languages.yaml", "r", encoding="utf-8") as f:
            lang_cfg = yaml.safe_load(f)
        lang_entry = lang_cfg.get("languages", {}).get(args.lang, {})
        if lang_entry.get("dialect_post_process", False):
            dialect_script = lang_entry["dialect_script"]
            transcript_path = f"data/processed/transcript_{args.lang}.json"
            run_stage(f"Stage 4b: Dialect Post-Processing ({args.lang})",
                      dialect_script,
                      ["--input", transcript_path, "--output", transcript_path])

    if args.start_stage <= 5:
        tts_args = ["--input", f"data/processed/transcript_{args.lang}.json",
                    "--output-dir", "data/tts_output/",
                    "--lang", args.lang]
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

    if args.video_output:
        if not args.face_video:
            print("\n⚠️  Warning: --video-output set but --face-video not provided. Skipping Stage 7.")
        else:
            video_out = f"outputs/final_{args.lang}_video.mp4"
            run_stage("Stage 7: Lip Sync",
                      "scripts/inference/lip_sync.py",
                      ["--input", args.face_video,
                       "--audio", f"outputs/final_{args.lang}_dub.wav",
                       "--output", video_out])
            print(f"\n🎉  Pipeline complete!")
            print(f"    Audio  → outputs/final_{args.lang}_dub.wav")
            print(f"    Video  → {video_out}\n")
            return

    print(f"\n🎉  Pipeline complete!")
    print(f"    Output → outputs/final_{args.lang}_dub.wav\n")


if __name__ == "__main__":
    main()
