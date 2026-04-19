"""
scripts/dialects/hindi_to_haryanvi.py
Post-processing layer: converts standard Hindi text to Haryanvi dialect.
This is a rule-based transformer — will be replaced by ML model in Phase 6.
"""

# Common Haryanvi substitutions
HARYANVI_SUBSTITUTIONS = {
    "मैं": "म्हैं",
    "हूँ": "सूं",
    "है": "सै",
    "हैं": "सैं",
    "क्या": "के",
    "नहीं": "न्हीं",
    "यहाँ": "यड़े",
    "वहाँ": "वड़े",
    "कहाँ": "कड़े",
    "अच्छा": "बढ़िया",
    "बहुत": "घणा",
    "थोड़ा": "थोड़ा सा",
    "आओ": "आज्या",
    "जाओ": "जा",
    "खाना": "रोटी",
    "पानी": "पाणी",
    "दूध": "दूध",
    "घर": "घर",
    "भाई": "भाई",
    "भाईसाहब": "भाईया",
}

HARYANVI_SUFFIXES = {
    "ता है": "से सै",
    "ती है": "सी सै",
    "ते हैं": "से सैं",
    "रहा है": "रया सै",
    "रही है": "री सै",
}


def hindi_to_haryanvi(text: str) -> str:
    result = text

    for hindi, haryanvi in HARYANVI_SUFFIXES.items():
        result = result.replace(hindi, haryanvi)

    for hindi, haryanvi in HARYANVI_SUBSTITUTIONS.items():
        result = result.replace(hindi, haryanvi)

    return result


def process_segments(segments: list) -> list:
    for seg in segments:
        if 'text_translated' in seg:
            seg['text_dialect'] = hindi_to_haryanvi(seg['text_translated'])
    return segments


if __name__ == "__main__":
    test_sentences = [
        "मैं यहाँ हूँ और खाना खा रहा है।",
        "क्या तुम वहाँ जाओ?",
        "बहुत अच्छा है यह।",
    ]
    print("Hindi → Haryanvi dialect conversion test:\n")
    for s in test_sentences:
        print(f"  HI: {s}")
        print(f"  HRY: {hindi_to_haryanvi(s)}\n")
