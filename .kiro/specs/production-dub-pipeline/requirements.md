# Requirements Document

## Introduction

This spec covers the upgrades and additions needed to transform the existing `anime-dub-ai` project into a production-level anime dubbing pipeline. The existing pipeline already handles audio extraction, speaker diarization, ASR transcription, translation, TTS synthesis, and audio mixing. This document focuses on what must be **added or upgraded** to meet production quality standards:

- Multi-track video output (original Japanese audio + Hindi dubbed audio, switchable in VLC/MX Player)
- SRT subtitle input path (bypass Whisper when subtitles are provided)
- Natural, non-literal Hindi dubbing dialogue (localization-aware translation)
- Per-character voice assignment with consistent voice identity across an episode
- Emotion-driven TTS prosody (pitch, speed, intensity variation per emotion)
- Demucs-based audio source separation (preserve BGM and SFX, replace only dialogue)
- Production-quality audio mixing with balanced levels
- A training data folder structure and fine-tuning pipeline for learning dubbing style

---

## Glossary

- **Pipeline**: The end-to-end system that takes a video file and produces a dubbed video file.
- **Muxer**: The component responsible for combining audio tracks and video into a single container file using FFmpeg.
- **Dubbing_Adapter**: The component that converts literal translated text into natural, localized Hindi dubbing dialogue.
- **Character_Voice_Mapper**: The component that assigns and maintains consistent TTS voice profiles per detected speaker.
- **Source_Separator**: The component (Demucs) that splits audio into vocals and background (BGM + SFX).
- **Emotion_Engine**: The component that reads emotion labels and adjusts TTS prosody parameters accordingly.
- **TTS_Engine**: The text-to-speech synthesis component (Coqui VITS / NVIDIA NeMo / Bark).
- **Training_Pipeline**: The set of scripts and data structures used to fine-tune TTS models on custom dubbing data.
- **SRT_Parser**: The component that reads `.srt` subtitle files and converts them into timed segment objects, with optional alignment offset correction.
- **Segment**: A single timed unit of dialogue with start time, end time, speaker ID, source text, translated text, emotion label, and emotion intensity.
- **Voice_Embedding_Clusterer**: The component that uses Resemblyzer voice embeddings to group audio segments into speaker clusters when SRT input is provided without speaker names.
- **Log_Writer**: The component that writes timestamped stage entries to the pipeline log file and updates `logs/current_stage.txt` in real-time.
- **Mixer**: The component responsible for combining TTS audio segments with the background stem into a final mixed audio file, applying LUFS normalization and gain control.
- **Validator**: The component that runs post-mux output validation using FFprobe to verify the integrity of the output file.

---

## Requirements

### Requirement 1: Multi-Track Video Output

**User Story:** As a user, I want the output to be a video file with both the original Japanese audio and the new Hindi dubbed audio as separate tracks, so that I can switch between them in VLC or MX Player.

#### Acceptance Criteria

1. WHEN the Pipeline completes processing, THE Muxer SHALL produce an `.mp4` or `.mkv` output file containing the original video stream, the original Japanese audio as track 1, and the Hindi dubbed audio as track 2.
2. THE Muxer SHALL set track metadata so that the original Japanese audio track is labeled `"Japanese"` and the Hindi dubbed audio track is labeled `"Hindi"`.
3. WHEN the output file is opened in a player that supports multi-track audio (e.g. VLC, MX Player), THE Muxer SHALL produce a file where both audio tracks are independently selectable.
4. THE Muxer SHALL use FFmpeg stream copy for the video and original audio tracks to avoid re-encoding and quality loss.
5. IF the input file is `.mp4`, THEN THE Muxer SHALL produce an `.mp4` output; IF the input file is `.mkv`, THEN THE Muxer SHALL produce an `.mkv` output.
6. THE Muxer SHALL invoke FFmpeg as a subprocess using explicit stream mapping: `-map 0:v -map 0:a -map 1:a -c:v copy -c:a:0 copy -c:a:1 aac -metadata:s:a:0 language=jpn -metadata:s:a:1 language=hin -disposition:a:0 default` for `.mp4` output; WHERE the output format is `.mkv`, THE Muxer SHALL use `-c:a:1 libopus` or `-c:a:1 aac` for the Hindi audio track.
7. WHEN muxing completes, THE Muxer SHALL verify that the output file exists and has a file size greater than 0 bytes, and SHALL raise an error if either condition is not met.
8. THE Muxer SHALL log the exact FFmpeg command used for each mux operation to the pipeline log file.

---

### Requirement 2: SRT Subtitle Input Path

**User Story:** As a user, I want to provide an optional `.srt` subtitle file so that the pipeline uses its timings directly instead of running Whisper ASR, saving time and improving accuracy.

#### Acceptance Criteria

1. WHEN an `.srt` file is provided as input, THE SRT_Parser SHALL parse it into a list of Segments with `start`, `end`, and `text` fields matching the subtitle entries.
2. WHEN an `.srt` file is provided, THE Pipeline SHALL skip the Whisper ASR stage and use the SRT_Parser output as the transcript.
3. WHEN an `.srt` file is NOT provided, THE Pipeline SHALL run Whisper ASR as the transcript source (existing behavior).
4. IF an `.srt` file is provided but contains zero valid entries, THEN THE SRT_Parser SHALL return an error and THE Pipeline SHALL fall back to Whisper ASR.
5. THE SRT_Parser SHALL handle both UTF-8 and UTF-8-BOM encoded `.srt` files without error.
6. FOR ALL valid `.srt` files, parsing then re-serializing the segments SHALL produce timecodes equivalent to the originals within 10 milliseconds (round-trip property).
7. THE SRT_Parser SHALL apply a configurable alignment correction offset (default: 0ms, range: -300ms to +300ms) to all parsed segment timecodes.
8. THE Pipeline SHALL accept a `--subtitle-offset` argument (in milliseconds) that is passed to the SRT_Parser.
9. WHEN `--subtitle-offset` is provided, THE SRT_Parser SHALL add the offset to both `start` and `end` times of every parsed segment, clamping to a minimum of 0ms.
10. THE SRT_Parser SHALL log the applied offset and the number of segments adjusted.

---

### Requirement 3: Natural Hindi Dubbing Dialogue (Localization)

**User Story:** As a viewer, I want the Hindi dialogue to sound like natural Indian dubbing — not a literal translation — so that it feels authentic and emotionally resonant.

#### Acceptance Criteria

1. WHEN a translated segment is processed, THE Dubbing_Adapter SHALL rewrite the text to use conversational Hindi phrasing rather than word-for-word translation.
2. THE Dubbing_Adapter SHALL adapt anime-specific slang and honorifics (e.g. "senpai", "nakama", "nani") into natural Indian equivalents or commonly understood Hindi terms.
3. THE Dubbing_Adapter SHALL shorten dialogue to fit within the original segment duration, targeting a character count no more than 1.3× the estimated speakable characters for that duration at a natural Hindi speaking rate of 14 characters per second.
4. THE Dubbing_Adapter SHALL preserve the emotional intent and meaning of the source dialogue.
5. WHEN a segment has `emotion` set to `"angry"` or `"excited"`, THE Dubbing_Adapter SHALL use exclamatory Hindi phrasing appropriate to that emotion.
6. THE Dubbing_Adapter SHALL NOT produce output containing untranslated Japanese characters (Hiragana, Katakana, Kanji).
7. THE Dubbing_Adapter SHALL compute the available duration for each segment as `segment.end - segment.start` and target a character count fitting a natural Hindi speech rate of 14 characters per second.
8. WHEN TTS output for a segment exceeds the available segment duration, THE Dubbing_Adapter SHALL time-stretch the audio using `librosa.effects.time_stretch` at a stretch ratio of up to 1.4× before mixing.
9. IF time-stretching alone is insufficient to fit the TTS output within the segment duration, THEN THE Dubbing_Adapter SHALL shorten the translated text and re-synthesize the segment, with a maximum of 1 retry attempt.

---

### Requirement 4: Per-Character Voice Assignment

**User Story:** As a viewer, I want each character to have a distinct, consistent Hindi voice throughout the episode, so that I can identify characters by their voice.

#### Acceptance Criteria

1. THE Character_Voice_Mapper SHALL assign a unique TTS voice profile to each unique speaker ID detected by the diarization stage.
2. WHEN a speaker ID is encountered for the first time in an episode, THE Character_Voice_Mapper SHALL select a voice profile from `configs/character_voices.yaml` that has not yet been assigned to another speaker in the same episode.
3. WHEN a speaker ID has already been assigned a voice profile, THE Character_Voice_Mapper SHALL use the same voice profile for all subsequent segments from that speaker.
4. THE Character_Voice_Mapper SHALL support at least 4 distinct voice profiles simultaneously within a single episode.
5. THE Character_Voice_Mapper SHALL persist the speaker-to-voice mapping for an episode in `data/voice_maps/{episode_id}_voice_map.json` so that the mapping survives pipeline restarts and is reused for the same episode.
6. IF the number of detected speakers exceeds the number of available voice profiles, THEN THE Character_Voice_Mapper SHALL cycle through profiles and log a warning.
7. WHEN pyannote diarization is available, THE Character_Voice_Mapper SHALL use pyannote speaker diarization as the primary method for assigning speaker IDs to segments.
8. WHEN an `.srt` file is provided as input without embedded speaker names, THE Voice_Embedding_Clusterer SHALL extract Resemblyzer voice embeddings for each SRT-timed segment and group segments into speaker clusters based on embedding similarity, producing speaker IDs equivalent to those from diarization.
9. WHERE batch processing is performed across multiple episodes of the same series, THE Character_Voice_Mapper SHALL accept an optional `--shared-voice-map` argument pointing to an existing voice map JSON, and SHALL reuse that mapping for consistent character voices across episodes.

---

### Requirement 5: Emotion-Driven TTS Prosody

**User Story:** As a viewer, I want the Hindi voices to express emotion naturally — varying in pitch, speed, and intensity — so that the dub feels alive rather than robotic.

#### Acceptance Criteria

1. WHEN a segment has `emotion_intensity` greater than 0.7, THE Emotion_Engine SHALL route synthesis to Bark TTS using an emotion-matched Hindi voice preset.
2. WHEN a segment has `emotion` set to `"angry"`, THE Emotion_Engine SHALL apply a speaking rate increase of 10–20% and a pitch shift of +2 semitones relative to the neutral baseline.
3. WHEN a segment has `emotion` set to `"sad"`, THE Emotion_Engine SHALL apply a speaking rate decrease of 10–15% and a pitch shift of -1 semitone relative to the neutral baseline.
4. WHEN a segment has `emotion` set to `"happy"` or `"excited"`, THE Emotion_Engine SHALL apply a speaking rate increase of 5–15% and a pitch shift of +1 semitone relative to the neutral baseline.
5. WHEN a segment has `emotion` set to `"neutral"`, THE Emotion_Engine SHALL synthesize at the baseline speaking rate and pitch with no modification.
6. THE Emotion_Engine SHALL read emotion labels from the segment JSON produced by `detect_emotion.py` and SHALL NOT require manual annotation.
7. THE TTS_Engine SHALL auto-detect the available backend at startup in the following priority order: (1) NVIDIA NeMo FastPitch+HiFiGAN if CUDA is available and `nemo_toolkit` is installed, (2) Coqui VITS Hindi model on GPU or CPU, (3) gTTS as a lightweight CPU-only fallback. THE TTS_Engine SHALL log which backend was selected.
8. WHEN selecting a TTS backend, THE TTS_Engine SHALL verify that Hindi language support is available for the candidate backend before confirming selection, and SHALL skip to the next priority level if Hindi is not supported.
9. THE Emotion_Engine SHALL detect emotion using a two-stage strategy: first applying rule-based detection from text patterns (exclamation marks, question marks, and Hindi keywords such as "nahi!", "kyu!", "bachao"), then optionally applying AI-based tagging via `detect_emotion.py` (transformers-based) as a secondary enrichment step.
10. THE Emotion_Engine SHALL assign each segment an `emotion` field from the set `{neutral, happy, angry, sad, excited, fearful}` and an `emotion_intensity` float in the range 0.0–1.0.
11. WHEN AI-based emotion tagging fails or is unavailable, THE Emotion_Engine SHALL fall back to the rule-based result and SHALL always produce a valid `emotion` label for every segment.

---

### Requirement 6: Audio Source Separation (Preserve BGM and SFX)

**User Story:** As a viewer, I want the background music and sound effects to be preserved in the dubbed output, so that the audio feels complete and cinematic.

#### Acceptance Criteria

1. THE Source_Separator SHALL use Demucs to split the extracted audio into a `vocals` stem and a `background` stem (containing BGM and SFX).
2. WHEN source separation completes, THE Pipeline SHALL use the `background` stem as the BGM input to the audio mixer, replacing any manually provided `--bgm` argument.
3. THE Source_Separator SHALL produce stems at 22050 Hz sample rate to match the rest of the pipeline.
4. WHEN the `background` stem is mixed with the Hindi dubbed dialogue, THE Mixer SHALL set the background level to 0.20 and the dialogue level to 0.85 (existing ratios), maintaining balanced output.
5. IF Demucs is not installed or fails, THEN THE Source_Separator SHALL log a warning and THE Pipeline SHALL fall back to using the full extracted audio as the BGM input.
6. THE Source_Separator SHALL process a 24-minute audio file in under 10 minutes on a machine with a CUDA-capable GPU.
7. AFTER Demucs separation completes, THE Source_Separator SHALL compute the Signal-to-Noise Ratio (SNR) of the vocals stem.
8. IF the vocals stem SNR is below 10 dB, THEN THE Source_Separator SHALL log a warning and THE Pipeline SHALL fall back to dialogue ducking mode: the full original audio is used as background, with volume reduced by 60% during dialogue segments and restored to 100% during non-dialogue segments.
9. THE Source_Separator SHALL write a `data/processed/separation_quality.json` file reporting the SNR value and whether fallback was triggered.

---

### Requirement 7: Production Audio Mixing

**User Story:** As a viewer, I want the final dubbed audio to have clean, balanced levels where voices blend naturally with the background, so that the output is broadcast-quality.

#### Acceptance Criteria

1. THE Mixer SHALL normalize the final mixed audio to -16 LUFS (stereo) using the `pyloudnorm` library before writing the output file.
2. Peak normalization to -1.0 dBFS SHALL be applied as a secondary safety ceiling AFTER LUFS normalization to prevent clipping.
3. THE Mixer SHALL log the measured integrated LUFS value before and after normalization to the pipeline log.
4. THE Mixer SHALL apply a 10 ms fade-in and 10 ms fade-out to each TTS segment before mixing to eliminate click artifacts at segment boundaries.
5. WHEN two TTS segments overlap in time, THE Mixer SHALL sum them without clipping by applying the normalization in Acceptance Criteria 1 and 2 after all segments are placed.
6. THE Mixer SHALL output audio at 22050 Hz, 16-bit PCM WAV format for the intermediate dubbed audio file.
7. THE Mixer SHALL produce a mixing report JSON file listing each segment's start time, end time, speaker ID, stretch ratio, and final gain applied.

---

### Requirement 8: Training Data Structure and Fine-Tuning Pipeline

**User Story:** As a developer, I want a `training/` folder with a defined structure for Japanese source and Hindi dubbed pairs, so that I can fine-tune TTS models to learn the dubbing style.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL define a `training/` folder structure containing `source/` (Japanese audio clips), `dubbed/` (Hindi dubbed audio clips), and `metadata.csv` (LJ-Speech format with text and file references).
2. THE Training_Pipeline SHALL include a `prepare_dataset.py` script that reads the `training/` folder and produces a dataset in LJ-Speech format compatible with Coqui VITS fine-tuning.
3. WHEN `prepare_dataset.py` is run, THE Training_Pipeline SHALL validate that each entry in `metadata.csv` has a corresponding audio file in `wavs/` and SHALL report any missing files.
4. THE Training_Pipeline SHALL support loading pre-existing Coqui VITS checkpoints as the base model for fine-tuning via the `--base-model` argument (already supported in `finetune_vits.py` — this requirement covers the dataset preparation side).
5. WHERE NVIDIA NeMo is available, THE Training_Pipeline SHALL provide a `finetune_nemo.py` script that fine-tunes a NeMo FastPitch or VITS model using the same LJ-Speech formatted dataset.
6. THE Training_Pipeline SHALL include a `dataset_stats.py` script that reports total audio duration, number of speakers, average segment length, and vocabulary coverage for a given dataset directory.

---

### Requirement 9: SRT Subtitle Output

**User Story:** As a user, I want the pipeline to generate an `.srt` subtitle file for the Hindi dubbed audio, so that viewers can read along or use it for accessibility.

#### Acceptance Criteria

1. WHEN the Pipeline completes, THE Pipeline SHALL produce an `.srt` file alongside the output video containing the Hindi dubbed dialogue with timecodes matching the dubbed audio segments.
2. THE SRT_Parser SHALL serialize Segment objects into valid `.srt` format with sequential index numbers, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timecodes, and UTF-8 encoding.
3. FOR ALL Segment lists, serializing to `.srt` then parsing back SHALL produce Segments with timecodes within 10 milliseconds of the originals (round-trip property).
4. THE Pipeline SHALL write the `.srt` file to the same directory as the output video file, with the same base filename and a `.hi.srt` extension.

---

### Requirement 10: API and UI Upgrades for Multi-Track Output

**User Story:** As a developer, I want the FastAPI endpoint and Streamlit UI to support the new multi-track video output, so that users can upload a video (and optional SRT) and download the dubbed `.mp4`/`.mkv`.

#### Acceptance Criteria

1. THE API SHALL accept an optional `srt_file` upload parameter on the `POST /dub` endpoint alongside the existing `file` and `lang` parameters.
2. WHEN a job completes successfully, THE API `GET /download/{job_id}` endpoint SHALL return the muxed video file (`.mp4` or `.mkv`) with `Content-Type: video/mp4` or `video/x-matroska` respectively.
3. THE API SHALL validate that uploaded files have a `.mp4`, `.mkv`, or `.avi` extension and SHALL return HTTP 422 with a descriptive error for unsupported formats.
4. THE Streamlit UI SHALL display a file uploader for the video file and a separate optional uploader for the `.srt` subtitle file.
5. WHEN a job is complete, THE Streamlit UI SHALL provide a download button for the muxed video file.
6. THE API SHALL expose a `GET /voices` endpoint that returns the list of available voice profiles from `configs/character_voices.yaml`.

---

### Requirement 12: Overlapping Speech Handling

**User Story:** As a developer, I want the pipeline to detect and resolve overlapping dialogue segments automatically, so that the mixed audio does not contain garbled or clipped speech.

#### Acceptance Criteria

1. THE Pipeline SHALL detect overlapping dialogue segments, defined as two consecutive segments where one segment's start time is before the previous segment's end time.
2. WHEN overlapping segments are detected and the overlap duration is less than 200ms, THE Pipeline SHALL trim the earlier segment's end time to the later segment's start time.
3. WHEN overlapping segments are detected and the overlap duration is 200ms or more, THE Pipeline SHALL prioritize the segment with higher `emotion_intensity` and discard the lower-priority segment, logging the discarded segment's ID, text, and timecodes.
4. THE Pipeline SHALL report the count of overlapping segments detected and resolved in the mixing report JSON.

---

### Requirement 13: Output Validation

**User Story:** As a developer, I want the pipeline to automatically validate the output file after muxing, so that corrupt or incomplete outputs are caught before delivery.

#### Acceptance Criteria

1. AFTER muxing completes, THE Pipeline SHALL run an output validation step using FFprobe (via subprocess) to verify the output file.
2. THE Validator SHALL confirm that the output file contains exactly 2 audio tracks.
3. THE Validator SHALL confirm that the video duration and Hindi audio track duration are within 2 seconds of each other.
4. THE Validator SHALL confirm that the Hindi audio track has language metadata set to `"hin"`.
5. THE Validator SHALL confirm that the output file size is greater than 0 bytes.
6. IF any validation check fails, THEN THE Validator SHALL log the specific failure, write the failure details to the pipeline log, and raise a `PipelineValidationError` with a descriptive message.
7. THE Validator SHALL write a `logs/output_validation_{episode_id}.json` file with the full FFprobe output and pass/fail status for each check.

---

### Requirement 14: Pipeline Progress Logging

**User Story:** As a developer, I want the pipeline to write structured log files during processing, so that I can monitor progress in real-time, inspect partial results, and diagnose failures.

#### Acceptance Criteria

1. WHEN the Pipeline starts processing an episode, THE Log_Writer SHALL create a log file at `logs/pipeline_{episode_id}_{timestamp}.log` and write a timestamped entry for each stage as it begins and completes.
2. FOR EACH pipeline stage, THE Log_Writer SHALL record: stage name, start time, end time, duration in seconds, input file path(s), output file path(s), and segment count processed.
3. WHILE a pipeline stage is executing, THE Log_Writer SHALL overwrite `logs/current_stage.txt` with a single line containing the current stage name, enabling UI polling for real-time status.
4. IF a pipeline stage raises an exception, THEN THE Log_Writer SHALL write the stage name, exception type, exception message, and full traceback to the log file before the Pipeline exits.
5. WHEN each pipeline stage completes successfully, THE Pipeline SHALL write intermediate output files to disk so that partial results are inspectable without running the full pipeline.
