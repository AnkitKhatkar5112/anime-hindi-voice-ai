"""
scripts/inference/generate_subtitles.py
Generates SRT subtitle file from translated segments JSON.
"""
import json
from pathlib import Path


def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments_json: str, output_srt: str, lang_key: str = "text_translated") -> str:
    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    lines = []
    idx = 1

    for seg in segments:
        text = seg.get(lang_key, seg.get('text', '')).strip()
        if not text:
            continue

        start_str = seconds_to_srt_time(seg['start'])
        end_str = seconds_to_srt_time(seg['end'])

        lines.append(str(idx))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(text)
        lines.append("")
        idx += 1

    Path(output_srt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[Subtitles] Generated {idx-1} subtitle entries → {output_srt}")
    return output_srt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/transcript_hi.json")
    parser.add_argument("--output", default="outputs/subtitles_hi.srt")
    parser.add_argument("--lang-key", default="text_translated")
    args = parser.parse_args()

    generate_srt(args.input, args.output, args.lang_key)
