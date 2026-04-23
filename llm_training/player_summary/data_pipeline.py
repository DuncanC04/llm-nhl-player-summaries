"""JSONL load, shuffle, and HF Dataset construction."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from llm_training.player_summary.prompts import create_prompt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_data_file():
    """Find the data file by trying multiple possible paths."""
    possible_paths = [
        _REPO_ROOT / "Data" / "out" / "aiTop10Stats_complete.jsonl",
        Path("Data") / "out" / "aiTop10Stats_complete.jsonl",
        Path("aiTop10Stats_complete.jsonl"),
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.absolute())

    raise FileNotFoundError(
        f"Could not find data file! Tried: {[str(p) for p in possible_paths]}\n"
        "Make sure the data file exists in one of these locations."
    )


def load_data(data_path=None):
    """Load JSONL data file."""
    if data_path is None:
        data_path = find_data_file()

    print(f"Loading data from: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} examples")
    if len(data) > 0:
        print("\nSample entry:")
        print(json.dumps(data[0], indent=2))

    return data


def shuffle_examples(data, shuffle_seed):
    """Shuffle example dicts (copy). shuffle_seed None skips shuffle."""
    if shuffle_seed is None:
        return list(data)
    shuffled = list(data)
    rng = random.Random(shuffle_seed)
    rng.shuffle(shuffled)
    return shuffled


def create_training_example(example, tokenizer, preset) -> dict:
    """Single SFT row using the preset's formatting."""
    prompt = create_prompt(example)
    text = preset.build_training_text(example, prompt, tokenizer)
    return {"text": text}


def prepare_datasets(data, train_split=0.9, tokenizer=None, preset=None):
    """Prepare train and validation datasets (data order must match train/val indexing)."""
    formatted_data = [create_training_example(ex, tokenizer, preset) for ex in data]

    print("\nSample formatted training example:")
    print(formatted_data[0]["text"])
    print(f"\nTotal examples: {len(formatted_data)}")

    df = pd.DataFrame(formatted_data)

    train_size = int(train_split * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
        }
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    return dataset_dict, train_size
