"""
scripts/inference/batch_process.py
Process an entire folder of anime episodes.
Usage: python scripts/inference/batch_process.py --input-dir episodes/ --lang hi
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


SUPPORTED_FORMATS = {'.mp4', '.mkv', '.avi', '.wav', '.mp3'}


def find_episodes(input_dir: str) -> list:
    p = Path(input_dir)
    episodes = sorted([
        f for f in p.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])
    return episodes


def is_already_processed(episode_path: Path, lang: str) -> bool:
    output = Path("outputs") / f"{episode_path.stem}_{lang}_dub.wav"
    return output.exists()


def process_episode(episode_path: Path, lang: str, bgm: str = None,
                    skip_diarize: bool = False) -> dict:
    output_name = f"{episode_path.stem}_{lang}_dub.wav"
    output_path = str(Path("outputs") / output_name)

    cmd = [
        sys.executable, "scripts/inference/run_pipeline.py",
        "--input", str(episode_path),
        "--lang", lang,
    ]
    if bgm:
        cmd += ["--bgm", bgm]
    if skip_diarize:
        cmd.append("--skip-diarize")

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "episode": episode_path.name,
        "lang": lang,
        "output": output_path if result.returncode == 0 else None,
        "success": result.returncode == 0,
        "error": result.stderr[-300:] if result.returncode != 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Anime Dubbing")
    parser.add_argument("--input-dir", required=True, help="Folder with episode files")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--bgm", default=None)
    parser.add_argument("--skip-processed", action="store_true",
                        help="Skip already processed episodes")
    parser.add_argument("--skip-diarize", action="store_true")
    parser.add_argument("--report", default="logs/batch_report.json")
    args = parser.parse_args()

    episodes = find_episodes(args.input_dir)
    print(f"\n🎌 Batch Dub AI | Found {len(episodes)} episodes | Lang: {args.lang.upper()}")

    if args.skip_processed:
        episodes = [e for e in episodes if not is_already_processed(e, args.lang)]
        print(f"   Skipping already processed. Remaining: {len(episodes)}")

    results = []
    Path("logs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    for episode in tqdm(episodes, desc="Processing episodes"):
        print(f"\n▶ {episode.name}")
        result = process_episode(episode, args.lang, args.bgm, args.skip_diarize)
        results.append(result)

        status = "✅" if result["success"] else "❌"
        print(f"{status} {episode.name}")

    success_count = sum(1 for r in results if r["success"])
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "lang": args.lang,
        "results": results
    }

    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ {success_count}/{len(results)} episodes processed successfully")
    print(f"📄 Report saved: {args.report}")


if __name__ == "__main__":
    main()
