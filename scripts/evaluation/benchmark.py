"""
Benchmark script for PESQ/STOI quality tracking across multiple audio pairs.

Usage:
    python -m scripts.evaluation.benchmark --pairs '[{"original": "a.wav", "dubbed": "b.wav"}]'
    python -m scripts.evaluation.benchmark --pairs pairs.json --output logs/my_report.json
"""
import argparse
import json
import os
import sys
from datetime import date

# Allow running as a script directly (python scripts/evaluation/benchmark.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.evaluation.evaluate_quality import evaluate


def load_pairs(pairs_arg: str) -> list:
    """Accept either a raw JSON string or a path to a JSON file."""
    if os.path.isfile(pairs_arg):
        with open(pairs_arg, "r") as f:
            return json.load(f)
    return json.loads(pairs_arg)


def run_benchmark(pairs: list) -> tuple[list, list]:
    results = []
    errors = []

    for i, pair in enumerate(pairs):
        original = pair.get("original")
        dubbed = pair.get("dubbed")
        try:
            metrics = evaluate(original, dubbed)
            results.append({
                "pair_index": i,
                "original": original,
                "dubbed": dubbed,
                **metrics,
            })
        except Exception as e:
            error_msg = f"Pair {i} ({original} / {dubbed}): {e}"
            print(f"[WARN] Skipping {error_msg}", file=sys.stderr)
            errors.append({"pair_index": i, "original": original, "dubbed": dubbed, "error": str(e)})

    return results, errors


def print_summary(results: list, errors: list) -> None:
    n = len(results)
    if n == 0:
        print("\nNo pairs evaluated successfully.")
        return

    avg_pesq = sum(r["pesq_score"] for r in results) / n
    avg_stoi = sum(r["stoi_score"] for r in results) / n

    print("\n" + "=" * 50)
    print("  BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"  Pairs evaluated : {n}  (skipped: {len(errors)})")
    print(f"  Avg PESQ        : {avg_pesq:.4f}  (range 1–4.5)")
    print(f"  Avg STOI        : {avg_stoi:.4f}  (range 0–1)")
    print("=" * 50)


def save_report(results: list, errors: list, output_path: str) -> None:
    n = len(results)
    avg_pesq = sum(r["pesq_score"] for r in results) / n if n else None
    avg_stoi = sum(r["stoi_score"] for r in results) / n if n else None

    report = {
        "date": str(date.today()),
        "pairs_evaluated": n,
        "pairs_skipped": len(errors),
        "avg_pesq": round(avg_pesq, 4) if avg_pesq is not None else None,
        "avg_stoi": round(avg_stoi, 4) if avg_stoi is not None else None,
        "results": results,
        "errors": errors,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PESQ/STOI across audio pairs")
    parser.add_argument(
        "--pairs",
        required=True,
        help='JSON string or path to JSON file: [{"original": "...", "dubbed": "..."}]',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: logs/benchmark_YYYY-MM-DD.json)",
    )
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)

    output_path = args.output or os.path.join("logs", f"benchmark_{date.today()}.json")

    results, errors = run_benchmark(pairs)
    print_summary(results, errors)
    save_report(results, errors, output_path)


if __name__ == "__main__":
    main()
