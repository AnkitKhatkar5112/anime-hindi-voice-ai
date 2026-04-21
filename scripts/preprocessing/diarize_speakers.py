"""
Stage 1b: Speaker Diarization — Who is speaking when?
Segments audio by speaker for per-character voice cloning.
Requires HuggingFace token with pyannote/speaker-diarization-3.1 access.
"""
from pyannote.audio import Pipeline
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def diarize(audio_path: str, output_json: str, hf_token: str) -> dict:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
            "duration": round(turn.end - turn.start, 3)
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(segments, f, indent=2)

    unique_speakers = len(set(s['speaker'] for s in segments))
    print(f"[Diarization] Found {unique_speakers} speakers, {len(segments)} segments")
    return segments


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="data/processed/diarization.json")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    diarize(args.audio, args.output, args.hf_token)
