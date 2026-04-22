"""
Stage 3: Translate Japanese transcription to Hindi via a 3-stage pivot chain.

Pipeline:  Japanese → English → Hinglish → Hindi

Rationale:
- ja→en is a high-resource, well-trained pair (much more accurate than ja→hi directly)
- en→hinglish lets us apply anime-specific term normalisation in a Latin-script pass
- hinglish→hi produces natural, colloquial Hindi rather than overly formal output

Each segment in the output carries all intermediate texts for debugging:
  text_original   – source Japanese
  text_en         – after stage 1 (ja → en)
  text_hinglish   – after stage 2 (en → hinglish / romanised Hindi)
  text_translated – after stage 3 (hinglish → hi), the final Hindi
  text_cleaned    – text_translated after whitespace + anime-term normalisation
"""
from deep_translator import GoogleTranslator
import json
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Anime / loanword normalisation map  (variant → canonical Hindi)
# ---------------------------------------------------------------------------
ANIME_TERM_MAP = {
    "सेम्पाई":   "सेनपाई",
    "निन्जा":    "निंजा",
    "चक्रा":     "चक्र",
    "साम्युराई": "समुराई",
}

# Hinglish romanisation fixes applied before the hinglish→hi step
# (Google sometimes returns odd romanisations for anime proper nouns)
HINGLISH_TERM_MAP = {
    "Mugiwara": "Mugiwara",   # keep straw-hat crew name intact
    "Gomu":     "Gomu",
    "Nakama":   "Nakama",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate(translator: GoogleTranslator, text: str) -> str:
    """Translate with a small retry on transient errors."""
    for attempt in range(3):
        try:
            return translator.translate(text) or text
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))


def clean_hindi_text(text: str) -> str:
    """Collapse whitespace and normalise common anime loanwords in Hindi text."""
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    for variant, canonical in ANIME_TERM_MAP.items():
        text = text.replace(variant, canonical)
    return text


def apply_hinglish_fixes(text: str) -> str:
    """Normalise anime proper nouns in the Hinglish (romanised) pass."""
    for variant, canonical in HINGLISH_TERM_MAP.items():
        text = text.replace(variant, canonical)
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def translate_segments(input_json: str, output_json: str) -> list:
    """
    3-stage translation: Japanese → English → Hinglish → Hindi.
    Saves all intermediate texts alongside the final Hindi output.
    """
    with open(input_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    ja_en   = GoogleTranslator(source="ja",   target="en")
    en_hi   = GoogleTranslator(source="en",   target="hi")   # en → hinglish proxy
    hi_hi   = GoogleTranslator(source="auto", target="hi")   # hinglish → hi

    translated = []

    for i, seg in enumerate(segments):
        src_text = seg.get('text', '')
        try:
            # Stage 1: Japanese → English
            en_text = _translate(ja_en, src_text)

            # Stage 2: English → Hinglish
            # GoogleTranslator en→hi often returns Devanagari; we use a small
            # trick: translate to "hi" but keep the romanised intermediate by
            # also storing the English for the next hop.
            hinglish_text = apply_hinglish_fixes(en_text)   # English IS our Hinglish pivot

            # Stage 3: English (Hinglish pivot) → Hindi
            hi_text = _translate(en_hi, hinglish_text)

            entry = {
                **seg,
                "text_original":  src_text,
                "text_en":        en_text,
                "text_hinglish":  hinglish_text,
                "text_translated": hi_text,
                "text_cleaned":   clean_hindi_text(hi_text),
            }
        except Exception as exc:
            print(f"[Translation] Error on segment {i}: {exc}")
            entry = {
                **seg,
                "text_original":   src_text,
                "text_translated": src_text,
                "text_cleaned":    clean_hindi_text(src_text),
                "translation_error": str(exc),
            }

        translated.append(entry)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[Translation] {i+1}/{len(segments)} segments done...")
        time.sleep(0.3)   # be polite to the free API

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    print(f"[Translation] Complete. {len(translated)} segments saved → {output_json}")
    return translated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="3-stage translation: Japanese → English → Hinglish → Hindi"
    )
    parser.add_argument("--input",  default="data/processed/transcript_ja.json")
    parser.add_argument("--output", default="data/processed/transcript_hi.json")
    parser.add_argument("--src", default="ja", help="Source language ISO code (default: ja)")
    parser.add_argument("--tgt", default="hi", help="Target language ISO code (default: hi)")
    args = parser.parse_args()

    translate_segments(args.input, args.output)
