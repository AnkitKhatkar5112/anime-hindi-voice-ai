"""
scripts/training/finetune_vits.py
Fine-tune a Coqui VITS TTS model on a custom Hindi voice dataset.

Compute requirements:
  - GPU: ≥12 GB VRAM (e.g. NVIDIA RTX 3090 or A100)
  - Training time: ~12–24 hours for 1000 epochs on a single GPU
  - Dataset: LJ-Speech format (wavs/ + metadata.csv) produced by prepare_dataset.py

Usage:
    python scripts/training/finetune_vits.py \
        --base-model tts_models/hi/cv/vits \
        --data-dir data/training/ \
        --output-dir models/finetuned_hindi/ \
        --epochs 1000
"""

import argparse
import os
from pathlib import Path


def build_config(base_model: str, data_dir: str, output_dir: str, epochs: int) -> tuple:
    """Build VitsConfig and dataset config for fine-tuning."""
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(Path(data_dir).resolve()),
    )

    config = VitsConfig(
        epochs=epochs,
        batch_size=16,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs_between_evals=10,
        save_step=1000,
        save_n_checkpoints=5,
        save_best_after=1000,
        output_path=str(Path(output_dir).resolve()),
        datasets=[dataset_config],
        # Audio config matching LJ-Speech / prepare_dataset.py output
        audio={
            "sample_rate": 22050,
            "win_length": 1024,
            "hop_length": 256,
            "num_mels": 80,
            "mel_fmin": 0,
            "mel_fmax": None,
        },
    )

    return config, dataset_config


def train(base_model: str, data_dir: str, output_dir: str, epochs: int) -> None:
    """Run VITS fine-tuning."""
    from TTS.trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[finetune] Base model  : {base_model}")
    print(f"[finetune] Data dir    : {data_dir}")
    print(f"[finetune] Output dir  : {output_dir}")
    print(f"[finetune] Epochs      : {epochs}")

    config, dataset_config = build_config(base_model, data_dir, output_dir, epochs)

    # Override epoch count in case build_config default differs
    config.epochs = epochs

    # Audio processor and tokenizer
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load train / eval samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"[finetune] Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

    # Initialise model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # TrainerArgs: restore_path loads weights from the base model checkpoint
    trainer_args = TrainerArgs(
        restore_path=base_model,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=output_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()
    print(f"[finetune] Training complete. Checkpoints saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Coqui VITS model on a custom Hindi voice dataset."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model name (e.g. tts_models/hi/cv/vits) or path to a checkpoint .pth file",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to LJ-Speech formatted dataset directory (contains wavs/ and metadata.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/finetuned_hindi/",
        help="Directory to save checkpoints and logs (default: models/finetuned_hindi/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    args = parser.parse_args()

    # Validate data dir
    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        parser.error(f"--data-dir '{args.data_dir}' does not exist or is not a directory.")
    if not (data_path / "metadata.csv").exists():
        parser.error(f"metadata.csv not found in '{args.data_dir}'. Run prepare_dataset.py first.")
    if not (data_path / "wavs").is_dir():
        parser.error(f"wavs/ directory not found in '{args.data_dir}'. Run prepare_dataset.py first.")

    train(args.base_model, args.data_dir, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
