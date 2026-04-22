"""
Compare base vs fine-tuned Hindi TTS model outputs.

Generates 10 Hindi sentences with both models, runs PESQ/STOI evaluation
on each pair, and prints a formatted comparison table.

Usage:
    python scripts/evaluation/compare_voices.py \
        --finetuned-model models/finetuned_hindi/best_model.pth \
        --finetuned-config models/finetuned_hindi/config.json \
        --output-dir outputs/eval/ \
        --reference-wav data/voice_references/speaker.wav
"""
import argparse
import os
import sys
import torch
from pathlib import Path

# fmt: off
HINDI_SENTENCES = [
    "नमस्ते, आप कैसे हैं?",
    "आज मौसम बहुत अच्छा है।",
    "मुझे हिंदी बोलना पसंद है।",
    "यह एक परीक्षण वाक्य है।",
    "कृपया ध्यान से सुनें।",
    "आपका स्वागत है।",
    "मैं एक एनीमे डबिंग प्रणाली हूँ।",
    "यह आवाज़ बहुत सुंदर लगती है।",
    "हम मिलकर काम करते हैं।",
    "धन्यवाद, फिर मिलेंगे।",
]
# fmt: on

BASE_MODEL = "tts_models/hi/cv/vits"


def generate_audio(tts, sentences: list[str], output_dir: Path, prefix: str) -> list[str]:
    """Generate audio for each sentence and return list of output paths."""
    paths = []
    for i, text in enumerate(sentences):
        out_path = str(output_dir / f"{prefix}_{i:02d}.wav")
        print(f"  [{prefix}] Sentence {i+1}/{len(sentences)}: {text[:40]}...")
        tts.tts_to_file(text=text, file_path=out_path)
        paths.append(out_path)
    return paths


def print_table(results: list[dict], sentences: list[str]) -> None:
    """Print a formatted PESQ/STOI comparison table."""
    col_w = 42
    print("\n" + "=" * 90)
    print(f"{'#':<4} {'Sentence':<{col_w}} {'Base PESQ':>10} {'FT PESQ':>10} {'Base STOI':>10} {'FT STOI':>10}")
    print("-" * 90)

    base_pesq_vals, ft_pesq_vals = [], []
    base_stoi_vals, ft_stoi_vals = [], []

    for i, r in enumerate(results):
        sentence = sentences[i][:col_w - 1]
        base_pesq = r["base"]["pesq_score"]
        ft_pesq   = r["finetuned"]["pesq_score"]
        base_stoi = r["base"]["stoi_score"]
        ft_stoi   = r["finetuned"]["stoi_score"]

        base_pesq_vals.append(base_pesq)
        ft_pesq_vals.append(ft_pesq)
        base_stoi_vals.append(base_stoi)
        ft_stoi_vals.append(ft_stoi)

        print(f"{i+1:<4} {sentence:<{col_w}} {base_pesq:>10.4f} {ft_pesq:>10.4f} {base_stoi:>10.4f} {ft_stoi:>10.4f}")

    print("-" * 90)
    avg_bp = sum(base_pesq_vals) / len(base_pesq_vals)
    avg_fp = sum(ft_pesq_vals)   / len(ft_pesq_vals)
    avg_bs = sum(base_stoi_vals) / len(base_stoi_vals)
    avg_fs = sum(ft_stoi_vals)   / len(ft_stoi_vals)
    print(f"{'AVG':<4} {'':<{col_w}} {avg_bp:>10.4f} {avg_fp:>10.4f} {avg_bs:>10.4f} {avg_fs:>10.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned Hindi TTS model quality."
    )
    parser.add_argument("--finetuned-model",  default=None, help="Path to fine-tuned .pth checkpoint")
    parser.add_argument("--finetuned-config", default=None, help="Path to fine-tuned model config.json")
    parser.add_argument("--output-dir",       default="outputs/eval/", help="Directory for generated WAV files")
    parser.add_argument("--reference-wav",    default=None, help="Reference speaker WAV (for subjective note only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[compare_voices] Using device: {device}")

    # --- Base model generation ---
    print(f"\n[1/3] Generating base model audio ({BASE_MODEL})...")
    from TTS.api import TTS
    base_tts = TTS(BASE_MODEL).to(device)
    base_paths = generate_audio(base_tts, HINDI_SENTENCES, output_dir, "base")

    # --- Fine-tuned model generation ---
    ft_paths = None
    if args.finetuned_model is None:
        print("\n[2/3] --finetuned-model not provided — skipping fine-tuned generation.")
        print("      Re-run with --finetuned-model <path> to include fine-tuned comparison.")
    else:
        print(f"\n[2/3] Generating fine-tuned model audio ({args.finetuned_model})...")
        ft_tts = TTS(
            model_path=args.finetuned_model,
            config_path=args.finetuned_config,
        ).to(device)
        ft_paths = generate_audio(ft_tts, HINDI_SENTENCES, output_dir, "finetuned")

    # --- Evaluation ---
    if ft_paths is None:
        print("\n[3/3] Skipping evaluation (no fine-tuned outputs).")
        print(f"\nBase model WAVs saved to: {output_dir}")
        return

    print("\n[3/3] Running PESQ/STOI evaluation on each pair...")
    # Suppress the per-pair print from evaluate_quality
    import io
    from contextlib import redirect_stdout
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_quality import evaluate

    results = []
    for i, (bp, fp) in enumerate(zip(base_paths, ft_paths)):
        print(f"  Evaluating pair {i+1}/{len(HINDI_SENTENCES)}...")
        with redirect_stdout(io.StringIO()):
            base_scores = evaluate(bp, fp)
            ft_scores   = evaluate(fp, bp)
        results.append({"base": base_scores, "finetuned": ft_scores})

    # --- Print table ---
    print_table(results, HINDI_SENTENCES)

    # --- Subjective note ---
    print("\n📢  Subjective Listening Test Reminder")
    print("    Play each base_XX.wav and finetuned_XX.wav pair back-to-back.")
    if args.reference_wav:
        print(f"    Reference speaker: {args.reference_wav}")
        print("    Note which version sounds closer to the reference speaker's voice,")
        print("    prosody, and naturalness.")
    else:
        print("    Note which version sounds more natural and speaker-consistent.")
        print("    Tip: pass --reference-wav <path> to include the target speaker for comparison.")
    print(f"\n    Output directory: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
