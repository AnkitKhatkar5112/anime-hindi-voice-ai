"""
scripts/training/extract_voice_embeddings.py
Extract speaker voice embeddings using Resemblyzer.
Used for per-character voice cloning in TTS stage.
"""
import json
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import librosa


def extract_speaker_segments(audio_path: str, diarization_json: str,
                              min_duration: float = 3.0) -> dict:
    with open(diarization_json, 'r') as f:
        diarization = json.load(f)

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    speaker_audio = {}
    for seg in diarization:
        spk = seg['speaker']
        start = int(seg['start'] * sr)
        end = int(seg['end'] * sr)
        if end - start < int(min_duration * sr):
            continue
        chunk = audio[start:end]
        if spk not in speaker_audio:
            speaker_audio[spk] = []
        speaker_audio[spk].append(chunk)

    return speaker_audio


def compute_embeddings(speaker_audio: dict, output_dir: str) -> dict:
    encoder = VoiceEncoder()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for speaker_id, chunks in speaker_audio.items():
        if not chunks:
            continue

        combined = np.concatenate(chunks)
        wav = preprocess_wav(combined, source_sr=16000)

        if len(wav) < 16000:
            print(f"[Embeddings] {speaker_id}: not enough audio, skipping")
            continue

        embedding = encoder.embed_utterance(wav)

        out_file = str(output_path / f"{speaker_id}.npy")
        np.save(out_file, embedding)

        total_secs = sum(len(c) for c in chunks) / 16000
        manifest[speaker_id] = {
            "embedding_file": out_file,
            "total_audio_seconds": round(total_secs, 2),
            "num_segments": len(chunks),
            "embedding_dim": embedding.shape[0]
        }
        print(f"[Embeddings] {speaker_id}: {total_secs:.1f}s audio → {out_file}")

    manifest_path = str(output_path / "speaker_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Embeddings] Done. {len(manifest)} speakers → {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/processed/audio.wav")
    parser.add_argument("--diarization", default="data/processed/diarization.json")
    parser.add_argument("--output-dir", default="data/voice_references/embeddings/")
    args = parser.parse_args()

    segments = extract_speaker_segments(args.audio, args.diarization)
    compute_embeddings(segments, args.output_dir)
