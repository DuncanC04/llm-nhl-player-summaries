"""Qwen3-1.7B text preset."""

from __future__ import annotations

import torch


class Qwen3_1_7BPreset:
    id = "qwen3-1.7b"
    default_model_id = "Qwen/Qwen3-1.7B"
    lora_target_candidates = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    )

    def build_training_text(self, example: dict, prompt: str, tokenizer) -> str:
        return f"{prompt} {example['summary']}"

    def build_generation_inputs(self, tokenizer, prompt: str):
        batch = tokenizer(prompt, return_tensors="pt")
        inputs = {k: batch[k] for k in batch if isinstance(batch[k], torch.Tensor)}
        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    def extra_generation_stop_markers(self) -> tuple[str, ...]:
        return ()
