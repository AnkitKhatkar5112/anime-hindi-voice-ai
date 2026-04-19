# Phase 6 — Model Fine-Tuning: Tasks

- [ ] 1. Dataset preparation for TTS training
  - [ ] 1.1 Create `scripts/training/prepare_dataset.py` for LJ-Speech format conversion
    - Accept `--audio-dir`, `--transcripts` (CSV/JSON), `--output-dir` argparse arguments
    - Per file: load with librosa → resample to 22050 Hz mono → trim silence (`top_db=30`)
    - Skip files shorter than 1s or longer than 15s
    - Compute RMS energy — skip files below threshold (too quiet)
    - Write to `output-dir/wavs/{id}.wav` + `metadata.csv` in LJ-Speech pipe-delimited format
    - _Requirements: 1_
  - [ ] 1.2 Add dataset summary and validation reporting
    - Print: total files processed/skipped, total duration in hours
    - Print duration histogram (files per 1-second bucket)
    - Flag warning if total duration < 2 hours
    - Document data sources in script header: IITM Hindi TTS, AIR corpus, custom recordings
    - **Done when:** Script runs on sample audio folder, produces valid `metadata.csv` + `wavs/`, prints summary
    - _Requirements: 1, 2, 4_

---

- [ ] 2. VITS model fine-tuning on custom Hindi voice
  - [ ] 2.1 Create `scripts/training/finetune_vits.py` for Coqui VITS fine-tuning
    - Use `Trainer`, `VitsConfig`, `load_tts_samples`, `Vits` from Coqui TTS
    - Load base model config, override: `epochs=1000`, `batch_size=16`, `eval_batch_size=16`
    - Set dataset formatter to `"ljspeech"` pointing to prepared data
    - Use `TrainerArgs(restore_path=...)` to initialize from base model weights
    - Add `--base-model`, `--data-dir`, `--output-dir`, `--epochs` argparse arguments
    - **Compute requirements:** GPU ≥12 GB VRAM (RTX 3090 / A100), ~12–24 hours for 1000 epochs
    - **Done when:** Training runs for 100+ steps without crashing; checkpoint saved to `models/finetuned_hindi/`
    - _Requirements: 2, 3, 5_

---

- [ ] 3. Model evaluation and pipeline integration
  - [ ] 3.1 Create `scripts/evaluation/compare_voices.py` for base vs fine-tuned comparison
    - Generate same 10 Hindi sentences with both base and fine-tuned model
    - Save outputs to `outputs/eval/base_*.wav` and `outputs/eval/finetuned_*.wav`
    - Run `evaluate_quality.py` on each pair, print PESQ/STOI comparison table
    - Also do subjective listening test: note which sounds more like the reference speaker
    - **Done when:** Comparison table printed; fine-tuned model subjectively closer to reference
    - _Requirements: 3_
  - [ ] 3.2 Update pipeline to use fine-tuned model in `scripts/inference/tts_hindi.py`
    - Add `finetuned_model_path` field to `configs/pipeline_config.yaml`
    - If config path is set, load fine-tuned model via `TTS(model_path=..., config_path=...)`
    - Otherwise fall back to base `tts_models/hi/cv/vits`
    - **Done when:** Full pipeline with fine-tuned config produces audio in the target speaker's voice
    - _Requirements: 3, 5_

---

- [ ] 4. Haryanvi TTS training plan (long-term)
  - [ ] 4.1 Write `docs/haryanvi_tts_plan.md` with training roadmap
    - **Data collection:** target 10+ hours — Mozilla Common Voice campaign, studio recordings, podcast/YouTube (with consent)
    - **Data format:** LJ-Speech format from Task 1
    - **Architecture:** VITS from scratch (same as Coqui Hindi base)
    - **Compute estimate:** 48–96 hours on single A100 for 10-hour dataset
    - **Quality evaluation:** MOS test with native Haryanvi speakers
    - **Done when:** Document written with realistic timeline and resource estimates
    - _Requirements: 3_
