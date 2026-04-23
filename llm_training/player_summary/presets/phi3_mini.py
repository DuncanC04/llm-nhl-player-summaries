"""Microsoft Phi-3-mini instruct: chat template for train and inference."""

from __future__ import annotations

import torch


class Phi3MiniPreset:
    id = "phi-3-mini"
    default_model_id = "microsoft/Phi-3-mini-4k-instruct"
    # Transformers 5.x has native Phi-3 support; the cached remote code has a RoPE
    # scaling bug (KeyError: 'type') so we disable trust_remote_code for the model.
    trust_remote_code = False
    lora_target_candidates = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "gate_up_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def build_training_text(self, example: dict, prompt: str, tokenizer) -> str:
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            raise ValueError("phi-3-mini preset requires a tokenizer with chat_template")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["summary"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def build_generation_inputs(self, tokenizer, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(enc, torch.Tensor):
            inputs = {"input_ids": enc}
            input_length = enc.shape[1]
        else:
            inputs = {k: enc[k] for k in enc if isinstance(enc[k], torch.Tensor)}
            input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    def extra_generation_stop_markers(self) -> tuple[str, ...]:
        return ("<|end|>", "<|user|>", "<|assistant|>")
