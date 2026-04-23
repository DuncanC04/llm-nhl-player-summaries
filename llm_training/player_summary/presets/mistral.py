"""Mistral / Llama-style causal LM: plain prompt + completion SFT."""

from __future__ import annotations

import torch


class MistralPreset:
    id = "mistral"
    default_model_id = "mistralai/Mistral-7B-v0.1"
    lora_target_candidates = ("q_proj", "k_proj", "v_proj", "o_proj")

    def build_training_text(self, example: dict, prompt: str, tokenizer) -> str:
        return f"{prompt} {example['summary']}"

    def build_generation_inputs(self, tokenizer, prompt: str):
        batch = tokenizer(prompt, return_tensors="pt")
        inputs = {k: batch[k] for k in batch if isinstance(batch[k], torch.Tensor)}
        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    def extra_generation_stop_markers(self) -> tuple[str, ...]:
        return ()
