# Spec: Phase 6 — Model Fine-Tuning

## Overview
Fine-tune the Hindi VITS model on custom voice actor recordings to produce character-specific voices that sound more natural and Indian than the base Common Voice model. Long-term goal: train a Haryanvi TTS model from scratch.

**Prerequisite:** Phase 1 and 2 complete. GPU with ≥12 GB VRAM strongly recommended.

---

## Requirements

1. Training data is formatted in LJ-Speech format (`metadata.csv` + `wavs/`)
2. At least 2 hours of clean, labeled Hindi speech per voice before fine-tuning
3. Fine-tuned model produces audio that matches the reference speaker's voice characteristics
4. A dataset preparation script validates audio quality and flags outliers
5. Training configuration is version-controlled in `configs/`

---

## Tasks

### Task 1: Create Dataset Preparation Script
Create `scripts/training/prepare_dataset.py`.

The script takes a folder of raw audio files + a transcript file and outputs LJ-Speech format.

```python
# Input:
#   --audio-dir: folder with .wav or .mp3 files
#   --transcripts: CSV or JSON with filename → text mapping
#   --output-dir: where to write wavs/ and metadata.csv
#
# Processing per file:
#   1. Load audio (librosa)
#   2. Resample to 22050 Hz mono
#   3. Trim leading/trailing silence with librosa.effects.trim(top_db=30)
#   4. Skip files shorter than 1 second or longer than 15 seconds
#   5. Compute RMS energy — skip files below threshold (too quiet)
#   6. Write to output-dir/wavs/{id}.wav
#
# Output metadata.csv format (LJ-Speech):
#   id|normalized_text|normalized_text
```

At the end, print a summary:
- Total files processed / skipped
- Total duration in hours
- Duration histogram (how many files per 1-second bucket)
- Flag if total duration < 2 hours with a warning

**Data sources to document in a comment at top of script:**
- IITM Hindi TTS: https://www.iitm.ac.in/donlab/tts/
- AIR corpus
- Custom recordings (preferred for character voices)

**Done when:** Script runs on a sample audio folder and produces valid `metadata.csv` + `wavs/`. Summary shows duration and outlier count.

---

### Task 2: Fine-Tune VITS on Custom Hindi Voice
Create `scripts/training/finetune_vits.py`.

```python
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig

# Load base model config
config = VitsConfig()
config.load_json("path/to/base_model_config.json")  # export from Coqui model

# Override for fine-tuning
config.output_path = "models/finetuned_hindi/"
config.run_name = "hindi_custom_voice_v1"
config.epochs = 1000
config.batch_size = 16
config.eval_batch_size = 16
config.num_loader_workers = 4
config.datasets = [{
    "formatter": "ljspeech",
    "path": "data/training/",
    "meta_file_train": "metadata.csv",
    "language": "hi",
}]

train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)
model = Vits.init_from_config(config)

trainer = Trainer(
    TrainerArgs(restore_path="path/to/base_model.pth"),
    config,
    output_path=config.output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
```

Add `--base-model`, `--data-dir`, `--output-dir`, and `--epochs` argparse arguments.

**Requirements:**
- Minimum 1 hour of speech for fine-tuning (2+ hours preferred)
- GPU with ≥12 GB VRAM (RTX 3090 / A100 recommended)
- Estimated training time: 12–24 hours for 1000 epochs

**Done when:** Training runs for at least 100 steps without crashing. Checkpoint saved to `models/finetuned_hindi/`.

---

### Task 3: Evaluate Fine-Tuned Model
After training, compare the fine-tuned model output against the base model.

Create `scripts/evaluation/compare_voices.py`:
- Generate the same 10 Hindi sentences with both the base and fine-tuned model
- Save outputs to `outputs/eval/base_*.wav` and `outputs/eval/finetuned_*.wav`
- Run `evaluate_quality.py` on each pair
- Print a comparison table: PESQ and STOI for base vs fine-tuned

Also do a subjective listening test: play both versions and note which sounds more like the reference speaker.

**Done when:** Comparison table printed. Fine-tuned model produces audio that subjectively matches the reference voice better than the base model.

---

### Task 4: Update Pipeline to Use Fine-Tuned Model
Edit `configs/pipeline_config.yaml` to add a `finetuned_model_path` field.

Edit `scripts/inference/tts_hindi.py` to load the fine-tuned model when this path is set:

```python
if config.get("finetuned_model_path"):
    tts = TTS(model_path=config["finetuned_model_path"],
              config_path=config["finetuned_config_path"]).to(device)
else:
    tts = TTS("tts_models/hi/cv/vits").to(device)
```

**Done when:** Running the full pipeline with the fine-tuned model config produces audio in the target speaker's voice.

---

### Task 5: Plan Haryanvi TTS Training (Long-Term)
Since no pre-trained Haryanvi TTS model exists, document the plan for training one from scratch.

Create `docs/haryanvi_tts_plan.md` covering:
1. **Data collection strategy:** target 10+ hours of clean Haryanvi speech
   - Mozilla Common Voice campaign for Haryanvi (requires community effort)
   - Studio recordings with native Haryanvi voice actors
   - Podcast/YouTube audio (with speaker consent + transcription)
2. **Data format:** same LJ-Speech format as Task 1
3. **Training architecture:** VITS from scratch (same as Coqui Hindi base)
4. **Estimated compute:** 48–96 hours on a single A100 for 10-hour dataset
5. **Quality evaluation plan:** MOS test with native Haryanvi speakers

**Done when:** `docs/haryanvi_tts_plan.md` written with realistic timeline and resource estimates.

---

## Acceptance Criteria

- [ ] `scripts/training/prepare_dataset.py` — runs on sample audio, produces valid LJ-Speech format
- [ ] `metadata.csv` — correct pipe-delimited format, all referenced wav files exist
- [ ] Dataset summary — prints total duration, flags if < 2 hours
- [ ] `scripts/training/finetune_vits.py` — runs for 100+ steps without crashing
- [ ] Checkpoint saved to `models/finetuned_hindi/`
- [ ] `scripts/evaluation/compare_voices.py` — prints PESQ/STOI comparison table
- [ ] Fine-tuned model subjectively closer to reference speaker than base model
- [ ] `configs/pipeline_config.yaml` — `finetuned_model_path` field wired into TTS script
- [ ] `docs/haryanvi_tts_plan.md` — written with data, compute, and timeline estimates
