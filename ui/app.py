"""
ui/app.py — Streamlit demo UI for Anime Dub AI
Run with: streamlit run ui/app.py
"""
import streamlit as st
import subprocess
import sys
import time
import tempfile
from pathlib import Path

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

STAGE_ORDER = [
    "Stage 1",
    "Stage 2",
    "Stage 3",
    "Stage 4",
    "Stage 5",
    "Stage 6",
]

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

        progress = st.progress(0)
        status_text = st.empty()
        status_text.text("Starting pipeline...")

        cmd = [sys.executable, "scripts/inference/run_pipeline.py",
               "--input", input_path, "--lang", lang_code, "--skip-diarize"]
        if bgm_path:
            cmd += ["--bgm", bgm_path]

        proc = subprocess.Popen(cmd)

        while proc.poll() is None:
            stage_file = Path("logs/current_stage.txt")
            if stage_file.exists():
                stage_label = stage_file.read_text(encoding="utf-8").strip()
                status_text.text(f"⏳ {stage_label}")
                # Compute progress fraction from stage number
                stage_num = 0
                for i, s in enumerate(STAGE_ORDER, start=1):
                    if stage_label.startswith(s):
                        stage_num = i
                        break
                if stage_num > 0:
                    progress.progress(stage_num / len(STAGE_ORDER))
            time.sleep(1)

        returncode = proc.returncode
        progress.progress(1.0)
        status_text.empty()

        if returncode == 0:
            output_file = Path(f"outputs/final_{lang_code}_dub.wav")
            if output_file.exists():
                st.success("✅ Dubbing complete!")
                audio_bytes = output_file.read_bytes()
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label="⬇️ Download Dubbed Audio",
                    data=audio_bytes,
                    file_name=f"dubbed_{lang_code}.wav",
                    mime="audio/wav"
                )
            else:
                st.error("Pipeline completed but output file not found.")
        else:
            st.error("❌ Pipeline failed.")

        with st.expander("📋 Show logs"):
            log_path = Path("logs/pipeline_stdout.txt")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                st.code("\n".join(lines[-20:]))

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
