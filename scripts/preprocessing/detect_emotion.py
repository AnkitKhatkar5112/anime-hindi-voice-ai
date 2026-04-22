"""
Stage 2b: Emotion Detection — Classify emotion for each transcript segment.
Uses j-hartmann/emotion-english-distilroberta-base from Hugging Face.
"""
import json
from pathlib import Path


# Map model output labels to canonical emotion set
EMOTION_MAP = {
    "anger":   "angry",
    "disgust": "neutral",
    "fear":    "fearful",
    "joy":     "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}


def detect_emotion(segments_json: str, output_json: str) -> list:
    from transformers import pipeline

    print("[Emotion] Loading classifier: j-hartmann/emotion-english-distilroberta-base")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        truncation=True,
    )

    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    for i, seg in enumerate(segments):
        text = seg.get('text_translated', seg.get('text', ''))
        if not text.strip():
            seg['emotion'] = 'neutral'
            seg['emotion_intensity'] = 0.0
            continue

        scores = classifier(text)[0]  # list of {label, score}
        top = max(scores, key=lambda x: x['score'])
        raw_label = top['label'].lower()
        seg['emotion'] = EMOTION_MAP.get(raw_label, 'neutral')
        seg['emotion_intensity'] = round(float(top['score']), 4)

        if i % 10 == 0:
            print(f"[Emotion] Processed {i+1}/{len(segments)} segments")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"[Emotion] Done. Output: {output_json}")
    return segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input segments JSON")
    parser.add_argument("--output", required=True, help="Path to output enriched JSON")
    args = parser.parse_args()

    detect_emotion(args.input, args.output)
