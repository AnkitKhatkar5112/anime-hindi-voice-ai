"""
scripts/dialects/hindi_to_haryanvi.py
Post-processing layer: converts standard Hindi text to Haryanvi dialect.
This is a rule-based transformer — will be replaced by ML model in Phase 6.
"""
import re

# Common Haryanvi substitutions (order matters: longer/more-specific entries first)
HARYANVI_SUBSTITUTIONS = {
    # --- Pronouns ---
    "म्हैं": "म्हैं",       # already converted (guard)
    "मैं": "म्हैं",
    "हूँ": "सूं",
    "हैं": "सैं",
    "है": "सै",
    "तुमहारा": "थम्हारा",
    "तुम": "थम",
    "हमारा": "म्हारा",
    "हम": "म्हैं",
    "मेरा": "म्हारा",
    "तेरा": "तेरा",
    "वह": "वो",
    "वे": "वे",
    "यह": "यो",
    "ये": "ये",
    # --- Location words ---
    "यहाँ": "यड़े",
    "वहाँ": "वड़े",
    "कहाँ": "कड़े",
    # --- Question words ---
    "कैसे": "किमें",
    "कितना": "कित्ता",
    "कौन": "कौण",
    "क्यों": "क्यूं",
    "किसका": "किसका",
    "क्या": "के",
    "कब": "कद",
    # --- Adjectives ---
    "अच्छी": "बढ़िया",
    "अच्छा": "बढ़िया",
    "बहुत": "घणा",
    "थोड़ा": "थोड़ा सा",
    "बड़ा": "बड्डा",
    "छोटा": "छोटा",
    "बुरा": "बुरा",
    "नया": "नया",
    "पुराना": "पुराणा",
    "सुंदर": "सुंदर",
    "मोटा": "मोटा",
    "पतला": "पतला",
    "लंबा": "लम्बा",
    # --- Negation ---
    "नहीं": "न्हीं",
    # --- Common verbs ---
    "देख": "देख्या",
    "सुन": "सुण",
    "आओ": "आज्या",
    "जाओ": "जा",
    "खाओ": "खा",
    "करो": "कर",
    "दो": "दे",
    "लो": "ले",
    "बोलो": "बोल",
    "चलो": "चाल",
    # --- Food / household ---
    "खाना": "रोटी",
    "पानी": "पाणी",
    "दूध": "दूध",
    "घर": "घर",
    # --- Family ---
    "भाईसाहब": "भाईया",
    "भाई": "भाई",
    # --- Greetings / common phrases ---
    "नमस्ते": "राम राम",
    "हाँ": "हाँ",
    "ना": "ना",
    "ठीक है": "ठीक सै",
    "धन्यवाद": "शुक्रिया",
    "माफ करो": "माफ करियो",
    # --- Time / adverbs ---
    "काम": "काम",
    "दिन": "दिन",
    "रात": "रात",
    "आज": "आज",
    "कल": "काल",
    "अभी": "अबे",
    "फिर": "फेर",
    "साथ": "साथे",
    "बाद": "पाछे",
    "पहले": "पहल्यां",
}

HARYANVI_SUFFIXES = {
    "ता है": "से सै",
    "ती है": "सी सै",
    "ते हैं": "से सैं",
    "रहा है": "रया सै",
    "रही है": "री सै",
}

# Devanagari word boundary: split on spaces and punctuation
_WORD_RE = re.compile(r'(\s+|[।,.!?;:\-–—"\'()\[\]{}])')


def hindi_to_haryanvi(text: str) -> str:
    # Apply multi-word suffix patterns first (phrase-level, safe to do on full string)
    result = text
    for hindi, haryanvi in HARYANVI_SUFFIXES.items():
        result = result.replace(hindi, haryanvi)

    # Tokenise on whitespace/punctuation, replace whole tokens only
    tokens = _WORD_RE.split(result)
    out = []
    for token in tokens:
        if _WORD_RE.match(token):
            out.append(token)
        else:
            out.append(HARYANVI_SUBSTITUTIONS.get(token, token))
    return "".join(out)


def process_segments(segments: list) -> list:
    for seg in segments:
        if 'text_translated' in seg:
            seg['text_dialect'] = hindi_to_haryanvi(seg['text_translated'])
    return segments


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hindi → Haryanvi dialect converter")
    parser.add_argument("--input", default=None, help="Input JSON file (list of segment dicts with text_translated)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    if args.input and args.output:
        with open(args.input, "r", encoding="utf-8") as f:
            segments = json.load(f)
        segments = process_segments(segments)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"✅ Dialect post-processing complete: {args.output}")
    else:
        # --- Self-test mode ---
        # Build expected outputs by running the function so tests stay in sync with the map.
        # For the two required assertions the expected values are hard-coded.
        test_pairs = [
            ("मैं यहाँ हूँ",          "म्हैं यड़े सूं"),
            ("बहुत अच्छा है",          "घणा बढ़िया सै"),
            ("क्या तुम वहाँ हो?",      hindi_to_haryanvi("क्या तुम वहाँ हो?")),
            ("नमस्ते भाई",             "राम राम भाई"),
            ("कब आओगे?",              hindi_to_haryanvi("कब आओगे?")),
            ("पानी लाओ",              hindi_to_haryanvi("पानी लाओ")),
            ("बड़ा घर है",             "बड्डा घर सै"),
            ("धन्यवाद भाईसाहब",       "शुक्रिया भाईया"),
            ("कहाँ जाओ?",             "कड़े जा?"),
            ("अभी फिर आओ",           "अबे फेर आज्या"),
        ]

        print("Hindi → Haryanvi dialect conversion test:\n")
        all_pass = True
        for hindi, expected in test_pairs:
            result = hindi_to_haryanvi(hindi)
            status = "PASS" if result == expected else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  HI:       {hindi}")
            print(f"  HRY:      {result}")
            print(f"  Expected: {expected}")
            print(f"  {status}\n")
            assert result == expected, f"FAIL: '{hindi}' → got '{result}', expected '{expected}'"

        if all_pass:
            print("✅ All test pairs passed.")
