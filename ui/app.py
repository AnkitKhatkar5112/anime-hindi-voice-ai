"""
ui/app.py — Streamlit demo UI for Anime Dub AI
Run with: streamlit run ui/app.py
"""
import streamlit as st
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import time

st.set_page_config(
    page_title="🎌 Anime Dub AI",
    page_icon="🎌",
    layout="centered"
)

st.title("🎌 Anime Voice Dubbing AI")
st.markdown("**Japanese anime → Natural Hindi (+ Indian language) voices**")
st.divider()

LANGUAGE_OPTIONS = {
    "Hindi (हिन्दी)": "hi",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
}

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Upload anime clip",
        type=["mp4", "mkv", "wav", "mp3"],
        help="Short clips (under 5 min) recommended for testing"
    )
with col2:
    selected_lang_name = st.selectbox("Target Language", list(LANGUAGE_OPTIONS.keys()))
    lang_code = LANGUAGE_OPTIONS[selected_lang_name]

bgm_file = st.file_uploader(
    "Background music (optional)",
    type=["wav", "mp3"],
    help="Original anime BGM/OST to mix in at low volume"
)

st.divider()

if uploaded_file and st.button("🚀 Start Dubbing", type="primary"):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        bgm_path = None
        if bgm_file:
            bgm_path = str(Path(tmpdir) / bgm_file.name)
            with open(bgm_path, "wb") as f:
                f.write(bgm_file.getvalue())

        stages = [
            "🎵 Extracting & cleaning audio...",
            "👥 Detecting speakers...",
            "🇯🇵 Transcribing Japanese (Whisper)...",
            f"🔄 Translating to {selected_lang_name}...",
            "🗣️ Synthesizing Hindi voices (TTS)...",
            "🎚️ Aligning timing & mixing...",
        ]

        progress = st.progress(0)
        status_text = st.empty()

        for i, stage in enumerate(stages):
            status_text.text(stage)
            progress.progress((i + 1) / len(stages))
            time.sleep(0.5)

        cmd = [sys.executable, "scripts/inference/run_pipeline.py",
               "--input", input_path, "--lang", lang_code, "--skip-diarize"]
        if bgm_path:
            cmd += ["--bgm", bgm_path]

        with st.spinner("Processing pipeline..."):
            result = subprocess.run(cmd, capture_output=True, text=True)

        progress.progress(1.0)

        if result.returncode == 0:
            output_file = f"outputs/final_{lang_code}_dub.wav"
            if Path(output_file).exists():
                st.success("✅ Dubbing complete!")
                st.audio(output_file)

                with open(output_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Dubbed Audio",
                        data=f,
                        file_name=f"dubbed_{lang_code}.wav",
                        mime="audio/wav"
                    )
            else:
                st.error("Pipeline completed but output file not found.")
        else:
            st.error("❌ Pipeline failed.")
            with st.expander("Error details"):
                st.code(result.stderr[-1000:])

st.divider()
st.markdown("""
**Pipeline Stages:**
1. 🎵 Audio extraction + noise reduction  
2. 👥 Speaker diarization (who speaks when)  
3. 🇯🇵 Japanese ASR via Whisper large-v3  
4. 🔄 Translation via Google Translate  
5. 🗣️ Hindi TTS via Coqui VITS  
6. 🎚️ Time alignment + background mix  
""")
