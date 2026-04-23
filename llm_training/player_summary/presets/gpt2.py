"""GPT-2 causal LM preset."""

from __future__ import annotations

import torch


class GPT2Preset:
    id = "gpt2"
    default_model_id = "openai-community/gpt2"
    lora_target_candidates = ("c_attn", "c_proj", "c_fc")

    def build_training_text(self, example: dict, prompt: str, tokenizer) -> str:
        return f"{prompt} {example['summary']}"

    def build_generation_inputs(self, tokenizer, prompt: str):
        batch = tokenizer(prompt, return_tensors="pt")
        inputs = {k: batch[k] for k in batch if isinstance(batch[k], torch.Tensor)}
        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    def extra_generation_stop_markers(self) -> tuple[str, ...]:
        return ()
