"""Argument parsing and train / test orchestration."""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore")

import torch

from llm_training.player_summary.data_pipeline import (
    load_data,
    prepare_datasets,
    shuffle_examples,
)
from llm_training.player_summary.evaluation_run import generate_all_summaries, test_model
from llm_training.player_summary.inference import load_finetuned_model
from llm_training.player_summary.model_setup import (
    load_tokenizer,
    setup_lora,
    setup_model_and_tokenizer,
)
from llm_training.player_summary.presets import (
    PRESET_IDS,
    default_model_for_preset,
    get_preset,
    resolve_lora_targets,
)
from llm_training.player_summary.training_loop import save_model, train_model


def check_cuda():
    """Check CUDA availability and print GPU information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    print("⚠️  WARNING: CUDA is not available. Training will be very slow on CPU.")
    return False


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared!")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for player summary generation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL data file (default: auto-detect)",
    )
    parser.add_argument(
        "--model_preset",
        choices=PRESET_IDS,
        default="mistral",
        help="Preset: default HF model id, loader type, LoRA targets, train/inference formatting (default: mistral)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face model id (default: from --model_preset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./player_summary_model",
        help="Directory to save the fine-tuned model (default: ./player_summary_model)",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Train/validation split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle JSONL rows before train/val split (default: 42). Ignored with --no_shuffle.",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Keep JSONL row order (no shuffle before split)",
    )
    parser.add_argument("--test_only", action="store_true", help="Only test an existing model (skip training)")
    parser.add_argument(
        "--generate_all",
        action="store_true",
        help="Generate summaries for all players after training/testing",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more memory)",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=3,
        help="Test examples after training (default: 3, -1 for all validation)",
    )
    parser.add_argument(
        "--export_predictions",
        type=str,
        default=None,
        help="Write validation predictions JSONL for evaluation.run_eval",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = default_model_for_preset(args.model_preset)
    preset = get_preset(args.model_preset)
    shuffle_seed = None if args.no_shuffle else args.shuffle_seed

    cuda_available = check_cuda()
    if not cuda_available:
        print("\n" + "=" * 80)
        print("ERROR: CUDA is not available!")
        print("=" * 80)
        print("\nThe advanced model requires a CUDA-enabled GPU.")
        print("\nYour system has an NVIDIA GPU, but PyTorch was installed")
        print("without CUDA support (CPU-only version).")
        print("\nTo fix this:")
        print("  1. Run: install_pytorch_cuda.bat (Windows) or install_pytorch_cuda.sh (Linux/Mac)")
        print("  2. Or manually:")
        print("     python -m pip uninstall -y torch torchvision torchaudio")
        print(
            "     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
        print("  3. Restart your terminal/Python environment")
        print("=" * 80)
        return

    if args.test_only:
        print("\n" + "=" * 80)
        print("TEST MODE: Loading fine-tuned model for inference")
        print("=" * 80)
        print("\nNOTE: Use the same --model_preset as training. Make sure the model has been trained first!")
        print("If you see poor quality output, the model may need more training.\n")

        data = load_data(args.data_path)
        data = shuffle_examples(data, shuffle_seed)
        if shuffle_seed is not None:
            print(f"Shuffled {len(data)} examples (seed={shuffle_seed}) before train/val split.")
        model, tokenizer = load_finetuned_model(args.output_dir, preset=preset)
        _, train_size = prepare_datasets(data, args.train_split, tokenizer, preset)
        test_model(
            model,
            tokenizer,
            data,
            train_size,
            num_test=args.num_test,
            export_predictions_path=args.export_predictions,
            preset=preset,
        )
        if args.generate_all:
            generate_all_summaries(data, model, tokenizer, preset=preset)
        cleanup_memory()
        return

    data = load_data(args.data_path)
    data = shuffle_examples(data, shuffle_seed)
    if shuffle_seed is not None:
        print(f"Shuffled {len(data)} examples (seed={shuffle_seed}) before train/val split.")

    tokenizer = load_tokenizer(args.model_name, preset=preset)
    dataset_dict, train_size = prepare_datasets(data, args.train_split, tokenizer, preset)

    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=not args.no_4bit,
        tokenizer=tokenizer,
        preset=preset,
    )

    lora_targets = resolve_lora_targets(model, preset.lora_target_candidates)
    print(f"LoRA target_modules: {lora_targets}")
    model, _peft_config = setup_lora(model, target_modules=lora_targets)

    trainer = train_model(
        model,
        tokenizer,
        dataset_dict["train"],
        dataset_dict["validation"],
        output_dir="./results",
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )

    save_model(trainer, tokenizer, args.output_dir)

    test_model(
        model,
        tokenizer,
        data,
        train_size,
        num_test=args.num_test,
        export_predictions_path=args.export_predictions,
        preset=preset,
    )

    if args.generate_all:
        generate_all_summaries(data, model, tokenizer, preset=preset)

    cleanup_memory()
    print("\nAll done.")
