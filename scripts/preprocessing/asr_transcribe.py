"""
Stage 2: ASR — Transcribe Japanese audio to text with word-level timestamps.
Uses faster-whisper for speed + accuracy on GPU.
"""
from faster_whisper import WhisperModel
import json
from pathlib import Path
import torch


def transcribe(audio_path: str, output_json: str, model_size: str = "large-v3", device: str = None) -> list:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[ASR] Using device: {device} | compute_type: {compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}
    )

    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in (seg.words or [])
            ]
        })

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[ASR] Transcribed {len(results)} segments | Language: {info.language} ({info.language_probability:.2f})")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="data/processed/transcript_ja.json")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default=None, help="Force device: cuda or cpu")
    args = parser.parse_args()

    transcribe(args.audio, args.output, args.model, args.device)
