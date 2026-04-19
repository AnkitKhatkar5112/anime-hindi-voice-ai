"""
Stage 3: Translate Japanese transcription to Hindi.
Preserves all segment timing metadata for downstream alignment.
"""
from deep_translator import GoogleTranslator
import json
import time
from pathlib import Path


def translate_segments(input_json: str, output_json: str,
                        src: str = "ja", tgt: str = "hi") -> list:
    with open(input_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    translator = GoogleTranslator(source=src, target=tgt)
    translated = []

    for i, seg in enumerate(segments):
        try:
            hi_text = translator.translate(seg['text'])
            translated.append({
                **seg,
                "text_original": seg['text'],
                "text_translated": hi_text,
                "target_lang": tgt
            })
            if i % 10 == 0:
                print(f"[Translation] {i+1}/{len(segments)} segments done...")
                time.sleep(0.4)
        except Exception as e:
            print(f"[Translation] Error on segment {i}: {e}")
            translated.append({**seg, "text_translated": seg['text'], "translation_error": str(e)})

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    print(f"[Translation] Complete. {len(translated)} segments → {tgt}")
    return translated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/transcript_ja.json")
    parser.add_argument("--output", default="data/processed/transcript_hi.json")
    parser.add_argument("--src", default="ja")
    parser.add_argument("--tgt", default="hi")
    args = parser.parse_args()

    translate_segments(args.input, args.output, args.src, args.tgt)
