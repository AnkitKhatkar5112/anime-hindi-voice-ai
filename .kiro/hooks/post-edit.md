# .kiro/hooks/post-edit.md
# Kiro automation hooks — run these checks automatically after editing pipeline scripts

## After editing any file in scripts/preprocessing/ or scripts/inference/

Run:
```bash
python -c "
import subprocess, sys

# Quick smoke test — extract + ASR on 30s fixture
scripts = [
    'scripts/preprocessing/extract_audio.py --input tests/fixtures/sample_30s.mp4 --output /tmp/smoke_audio.wav',
    'scripts/preprocessing/asr_transcribe.py --audio /tmp/smoke_audio.wav --output /tmp/smoke_ja.json --model tiny',
]
for cmd in scripts:
    r = subprocess.run([sys.executable] + cmd.split(), capture_output=True)
    print('OK' if r.returncode == 0 else f'FAIL: {cmd}')
"
```

## After editing configs/pipeline_config.yaml or configs/languages.yaml

Run:
```bash
python -c "
import yaml
for f in ['configs/pipeline_config.yaml', 'configs/languages.yaml', 'configs/character_voices.yaml']:
    with open(f) as fp:
        yaml.safe_load(fp)
    print(f'OK: {f}')
"
```

## Before committing

Run:
```bash
pytest tests/ -v --tb=short -q
```

## After adding a new language to configs/languages.yaml

Verify the pipeline accepts it without code changes:
```bash
python scripts/inference/run_pipeline.py --input tests/fixtures/sample_30s.mp4 --lang <new_lang> --start-stage 3
```
(Starting at stage 3 skips extraction/diarization to save time.)
