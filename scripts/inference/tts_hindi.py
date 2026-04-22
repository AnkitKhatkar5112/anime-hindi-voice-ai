"""
Stage 4: Hindi TTS — Convert translated Hindi text to speech.
Supports Coqui TTS (VITS model) with fallback to gTTS (Google TTS).
High-emotion segments (emotion_intensity > 0.7) are synthesized with Bark TTS.
"""
import json
import os
import torch
import numpy as np
import librosa as _lb
import soundfile as sf
from pathlib import Path

BARK_VOICE_MAP = {
    "angry":     "v2/hi_speaker_3",
    "happy":     "v2/hi_speaker_1",
    "sad":       "v2/hi_speaker_5",
    "surprised": "v2/hi_speaker_2",
    "fearful":   "v2/hi_speaker_4",
    "neutral":   "v2/hi_speaker_0",
}


def _try_load_coqui(device: str, model_name: str = "tts_models/hi/cv/vits",
                    finetuned_model_path: str = "", finetuned_config_path: str = ""):
    """Attempt to load a Coqui TTS model. Returns None if unavailable.

    If finetuned_model_path is set, loads the fine-tuned model from disk;
    otherwise falls back to the named base model.
    """
    try:
        from TTS.api import TTS
        if finetuned_model_path:
            print(f"[TTS] Loading fine-tuned model from: {finetuned_model_path}")
            tts = TTS(model_path=finetuned_model_path, config_path=finetuned_config_path).to(device)
        else:
            tts = TTS(model_name).to(device)
        return tts
    except (KeyError, Exception) as e:
        label = finetuned_model_path if finetuned_model_path else model_name
        print(f"[TTS] Coqui model '{label}' unavailable ({e}), falling back to gTTS")
        return None


def _bark_synthesize(text: str, emotion: str, out_file: str, segment_index: int = 0, intensity: float = 0.0):
    """Synthesize using Bark TTS with an emotion-matched Hindi voice preset."""
    from bark import generate_audio, SAMPLE_RATE
    prompt = BARK_VOICE_MAP.get(emotion, BARK_VOICE_MAP["neutral"])
    print(f"[TTS] Segment {segment_index}: using Bark (emotion={emotion}, intensity={intensity:.2f})")
    audio_array = generate_audio(text, history_prompt=prompt)
    sf.write(out_file, audio_array, SAMPLE_RATE)


def _gtts_synthesize(text: str, out_file: str):
    """Synthesize using Google TTS and save as WAV at 22050 Hz."""
    _gtts_synthesize_lang(text, out_file, lang='hi')


def _gtts_synthesize_lang(text: str, out_file: str, lang: str = 'hi'):
    """Synthesize using Google TTS for the given language and save as WAV at 22050 Hz."""
    from gtts import gTTS
    mp3_tmp = out_file.replace(".wav", "_tmp.mp3")
    gTTS(text, lang=lang).save(mp3_tmp)
    audio_data, _ = _lb.load(mp3_tmp, sr=22050, mono=True)
    sf.write(out_file, audio_data, 22050)
    os.remove(mp3_tmp)


def _find_speaker(seg_start: float, seg_end: float, diarization: list) -> str | None:
    """Return the speaker label whose diarization window best covers the segment midpoint."""
    midpoint = (seg_start + seg_end) / 2.0
    best_speaker = None
    best_overlap = -1.0
    for entry in diarization:
        d_start = entry['start']
        d_end = entry['end']
        # Use overlap length; fall back to proximity to midpoint
        overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = entry['speaker']
        elif overlap == 0 and best_overlap == 0:
            # No overlap found yet — pick closest by midpoint distance
            dist = abs(midpoint - (d_start + d_end) / 2.0)
            if best_speaker is None or dist < abs(midpoint - (diarization[0]['start'] + diarization[0]['end']) / 2.0):
                best_speaker = entry['speaker']
    return best_speaker


def synthesize_hindi(segments_json: str, output_dir: str,
                     engine: str = "coqui", speaker_embed: str = None,
                     device: str = None,
                     diarization_json: str = None,
                     lang: str = "hi",
                     finetuned_model_path: str = "",
                     finetuned_config_path: str = "") -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    # Load diarization data if provided
    diarization = None
    if diarization_json is not None:
        with open(diarization_json, 'r', encoding='utf-8') as f:
            diarization = json.load(f)

    tts = None
    active_engine = engine

    # Read languages.yaml for tts_engine override and tts_model path
    coqui_model = "tts_models/hi/cv/vits"  # default fallback
    try:
        import yaml
        with open("configs/languages.yaml", "r", encoding="utf-8") as _f:
            _lang_cfg = yaml.safe_load(_f)
        _lang_entry = _lang_cfg.get("languages", {}).get(lang, {})
        if _lang_entry.get("tts_engine") == "google":
            active_engine = "gtts"
        if _lang_entry.get("tts_model"):
            coqui_model = _lang_entry["tts_model"]
    except Exception:
        pass

    # Read pipeline_config.yaml for fine-tuned model paths (override CLI args if set)
    try:
        import yaml as _yaml
        with open("configs/pipeline_config.yaml", "r", encoding="utf-8") as _pf:
            _pipeline_cfg = _yaml.safe_load(_pf)
        _tts_cfg = _pipeline_cfg.get("tts", {})
        if not finetuned_model_path:
            finetuned_model_path = _tts_cfg.get("finetuned_model_path", "") or ""
        if not finetuned_config_path:
            finetuned_config_path = _tts_cfg.get("finetuned_config_path", "") or ""
    except Exception:
        pass

    if active_engine != "gtts" and engine == "coqui":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[TTS] Using device: {device}")
        tts = _try_load_coqui(device, coqui_model, finetuned_model_path, finetuned_config_path)
        if tts is None:
            active_engine = "gtts"

    if active_engine == "gtts":
        print(f"[TTS] Using engine: gTTS ({lang})")

    audio_segments = []

    for i, seg in enumerate(segments):
        text = seg.get('text_translated', seg.get('text', ''))
        if not text.strip():
            continue

        out_file = str(output_path / f"seg_{i:04d}.wav")

        # Resolve speaker from diarization
        speaker_id = None
        if diarization is not None:
            speaker_id = _find_speaker(
                seg.get('start', 0.0),
                seg.get('end', 0.0),
                diarization
            )
            # Check if a .npy embedding exists for this speaker
            embed_path = Path(f"data/voice_references/embeddings/{speaker_id}.npy")
            if speaker_id and embed_path.exists():
                print(f"[TTS] Segment {i}: speaker={speaker_id} (embedding found, logged only)")
            elif speaker_id:
                print(f"[TTS] Segment {i}: speaker={speaker_id} (no embedding file)")

        emotion = seg.get("emotion", "neutral")
        intensity = seg.get("emotion_intensity", 0.0)
        use_bark = intensity > 0.7

        if use_bark:
            _bark_synthesize(text, emotion, out_file, segment_index=i, intensity=intensity)
            tts_engine = "bark"
        elif active_engine == "coqui" and tts is not None:
            tts.tts_to_file(
                text=text,
                file_path=out_file,
                speaker_wav=speaker_embed
            )
            tts_engine = "coqui"
        elif active_engine == "gtts":
            _gtts_synthesize_lang(text, out_file, lang=lang)
            tts_engine = "gtts"
        else:
            _gtts_synthesize_lang(text, out_file, lang=lang)
            tts_engine = "gtts"

        # Compute stretch ratio metadata
        tts_audio, sr_ = _lb.load(out_file, sr=22050)
        tts_dur = len(tts_audio) / sr_
        orig_dur = seg['end'] - seg['start']
        seg['tts_duration'] = round(tts_dur, 3)
        seg['original_duration'] = round(orig_dur, 3)
        seg['stretch_ratio'] = round(tts_dur / orig_dur, 3) if orig_dur > 0 else 1.0

        audio_segments.append({
            **seg,
            "audio_file": out_file,
            "segment_index": i,
            "speaker_id": speaker_id,
            "tts_engine": tts_engine,
        })

        if i % 5 == 0:
            print(f"[TTS] Synthesized {i+1}/{len(segments)} segments")

    manifest_path = str(output_path / "segments.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(audio_segments, f, indent=2, ensure_ascii=False)

    print(f"[TTS] Done. Manifest: {manifest_path}")
    return audio_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/transcript_hi.json")
    parser.add_argument("--output-dir", default="data/tts_output/")
    parser.add_argument("--engine", default="coqui")
    parser.add_argument("--speaker-wav", default=None, help="Reference wav for voice cloning")
    parser.add_argument("--device", default=None, help="Device override: 'cuda' or 'cpu' (auto-detected if not set)")
    parser.add_argument("--diarization-json", default=None, help="Path to diarization JSON for per-speaker voice embedding lookup")
    parser.add_argument("--lang", default="hi", help="Language ISO code for gTTS fallback (e.g. hi, pa)")
    parser.add_argument("--finetuned-model", default="", help="Path to fine-tuned .pth model file")
    parser.add_argument("--finetuned-config", default="", help="Path to fine-tuned config.json")
    args = parser.parse_args()

    synthesize_hindi(args.input, args.output_dir, args.engine, args.speaker_wav, args.device, args.diarization_json, args.lang,
                     finetuned_model_path=args.finetuned_model, finetuned_config_path=args.finetuned_config)
