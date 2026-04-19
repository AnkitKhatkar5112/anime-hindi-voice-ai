"""
Stage 4: Hindi TTS — Convert translated Hindi text to speech.
Supports Coqui TTS (VITS model) with optional speaker embedding for voice cloning.
"""
from TTS.api import TTS
import soundfile as sf
import json
import os
from pathlib import Path


def synthesize_hindi(segments_json: str, output_dir: str,
                     engine: str = "coqui", speaker_embed: str = None) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    if engine == "coqui":
        tts = TTS("tts_models/hi/cv/vits").to("cuda")

    audio_segments = []

    for i, seg in enumerate(segments):
        text = seg.get('text_translated', seg.get('text', ''))
        if not text.strip():
            continue

        out_file = str(output_path / f"seg_{i:04d}.wav")

        if engine == "coqui":
            tts.tts_to_file(
                text=text,
                file_path=out_file,
                speaker_wav=speaker_embed
            )

        audio_segments.append({
            **seg,
            "audio_file": out_file,
            "segment_index": i
        })

        if i % 5 == 0:
            print(f"[TTS] Synthesized {i+1}/{len(segments)} segments")

    manifest_path = str(output_path / "segments.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(audio_segments, f, indent=2, ensure_ascii=False)

    print(f"[TTS] Done. Manifest: {manifest_path}")
    return audio_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/transcript_hi.json")
    parser.add_argument("--output-dir", default="data/tts_output/")
    parser.add_argument("--engine", default="coqui")
    parser.add_argument("--speaker-wav", default=None, help="Reference wav for voice cloning")
    args = parser.parse_args()

    synthesize_hindi(args.input, args.output_dir, args.engine, args.speaker_wav)
