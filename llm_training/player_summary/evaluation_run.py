"""Validation spot-check and full-dataset JSONL export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from evaluation.jsonl_table import stable_example_id
from llm_training.player_summary.inference import generate_player_summary
from llm_training.player_summary.prompts import format_stats_text


def _text_tokenizer(tokenizer_or_processor):
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def test_model(
    model,
    tokenizer,
    data,
    train_size,
    num_test=3,
    export_predictions_path=None,
    preset=None,
):
    """Test the model on validation examples."""
    print("\nTesting model on validation examples:\n")
    print("=" * 80)

    model.eval()

    val_set_size = len(data) - train_size
    if num_test > val_set_size:
        num_test = val_set_size
        print(f"Note: Only {val_set_size} examples in validation set, testing on all of them.\n")
    elif num_test == -1:
        num_test = val_set_size
        print(f"Testing on all {val_set_size} validation examples.\n")

    if num_test == 1:
        test_indices = [0]
    else:
        step = max(1, val_set_size // num_test)
        test_indices = list(range(0, val_set_size, step))[:num_test]

    print(f"Testing on {len(test_indices)} players from validation set:\n")

    generation_times = []
    export_rows = []

    for i, idx in enumerate(test_indices, 1):
        if idx < val_set_size:
            example = data[train_size + idx]

            print(f"\n[{i}/{len(test_indices)}] Player: {example['name']}")
            print(f"Team: {example['team']} | Position: {example['position']}")
            print(f"Top Stats: {format_stats_text(example['topStats'])}")

            print("\n--- ORIGINAL SUMMARY ---")
            print(example["summary"])

            print("\n--- GENERATED SUMMARY ---")
            generated = ""
            gen_time = 0.0
            num_out = 0
            peak_mb = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                generated, gen_time = generate_player_summary(
                    example["name"],
                    example["team"],
                    example["position"],
                    example["topStats"],
                    model,
                    tokenizer,
                    preset,
                    return_timing=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

                generation_times.append(gen_time)
                num_out = len(_text_tokenizer(tokenizer).encode(generated or "", add_special_tokens=False))

                if generated:
                    print(generated)
                    print(f"\n[Generation time: {gen_time:.3f} seconds]")
                else:
                    print("[WARNING: Empty generation - model may need more training]")
            except Exception as e:
                print(f"[ERROR generating summary: {e}]")

            if export_predictions_path:
                row = {
                    "id": stable_example_id(example),
                    "generated": generated or "",
                    "generation_time_s": float(gen_time),
                    "num_output_tokens": float(num_out),
                }
                if peak_mb is not None:
                    row["peak_gpu_mb"] = peak_mb
                export_rows.append(row)

            print("=" * 80)

    if export_predictions_path and export_rows:
        out_p = Path(export_predictions_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as ef:
            for row in export_rows:
                ef.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nExported {len(export_rows)} validation predictions for eval to: {out_p.resolve()}")

    if generation_times:
        print("\n" + "=" * 80)
        print("GENERATION TIME STATISTICS")
        print("=" * 80)
        print(f"Number of summaries generated: {len(generation_times)}")
        print(f"Average time per summary: {sum(generation_times)/len(generation_times):.3f} seconds")
        print(f"Min time: {min(generation_times):.3f} seconds")
        print(f"Max time: {max(generation_times):.3f} seconds")
        print(f"Total time: {sum(generation_times):.3f} seconds")
        print("=" * 80)


def generate_all_summaries(
    data,
    model,
    tokenizer,
    output_file="generated_summaries.jsonl",
    preset=None,
):
    """Generate summaries for all players and save to file."""
    results = []
    generation_times = []

    total_start_time = time.time()

    for i, example in enumerate(data):
        if i % 10 == 0:
            print(f"Processing {i}/{len(data)}...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        generated_summary, gen_time = generate_player_summary(
            example["name"],
            example["team"],
            example["position"],
            example["topStats"],
            model,
            tokenizer,
            preset,
            return_timing=True,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mb = None

        generation_times.append(gen_time)
        num_out = len(_text_tokenizer(tokenizer).encode(generated_summary or "", add_special_tokens=False))

        result = {
            "id": stable_example_id(example),
            "generated": generated_summary,
            "generation_time_s": gen_time,
            "num_output_tokens": num_out,
            "name": example["name"],
            "team": example["team"],
            "position": example["position"],
            "topStats": example["topStats"],
            "original_summary": example["summary"],
            "generated_summary": generated_summary,
        }
        if peak_mb is not None:
            result["peak_gpu_mb"] = float(peak_mb)
        results.append(result)

    total_time = time.time() - total_start_time

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nAll summaries generated and saved to: {output_file}")

    print("\n" + "=" * 80)
    print("BATCH GENERATION TIME STATISTICS")
    print("=" * 80)
    print(f"Total players processed: {len(data)}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per summary: {sum(generation_times)/len(generation_times):.3f} seconds")
    print(f"Min time: {min(generation_times):.3f} seconds")
    print(f"Max time: {max(generation_times):.3f} seconds")
    print(f"Summaries per minute: {60 / (sum(generation_times)/len(generation_times)):.1f}")
    print("=" * 80)

    return results
