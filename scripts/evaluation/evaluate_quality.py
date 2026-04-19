"""
Quality Evaluation — measures perceptual quality and intelligibility
of the final dubbed audio against a reference.
Metrics: PESQ, STOI, RMS Energy
"""
from pesq import pesq
from pystoi import stoi
import librosa
import numpy as np
import json
import argparse


def evaluate(original_path: str, dubbed_path: str, sr: int = 16000) -> dict:
    orig, _ = librosa.load(original_path, sr=sr)
    dub, _  = librosa.load(dubbed_path, sr=sr)

    min_len = min(len(orig), len(dub))
    orig, dub = orig[:min_len], dub[:min_len]

    pesq_score = pesq(sr, orig, dub, 'wb')
    stoi_score = stoi(orig, dub, sr, extended=False)

    results = {
        "pesq_score":            round(float(pesq_score), 4),
        "stoi_score":            round(float(stoi_score), 4),
        "rms_energy_original":   round(float(np.sqrt(np.mean(orig**2))), 6),
        "rms_energy_dubbed":     round(float(np.sqrt(np.mean(dub**2))), 6),
        "duration_original_s":   round(len(orig) / sr, 2),
        "duration_dubbed_s":     round(len(dub) / sr, 2),
    }

    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original audio (reference)")
    parser.add_argument("--dubbed", required=True, help="Dubbed output to evaluate")
    parser.add_argument("--output-json", default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = evaluate(args.original, args.dubbed)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output_json}")
