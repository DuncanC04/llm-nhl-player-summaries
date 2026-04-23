"""Merged adapter load, post-processing, and batched generation."""

from __future__ import annotations

import os
import re
import tempfile
import time
from difflib import SequenceMatcher

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from llm_training.player_summary.prompts import player_fields_prompt

_BASE_STOP_MARKERS = (
    "\n\n",
    "\nSummary:",
    "Summary:",
    "###",
    "---",
)


def _text_tokenizer(tokenizer_or_processor):
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def load_finetuned_model(model_dir="./player_summary_model", preset=None):
    """Load a fine-tuned model for inference."""
    print(f"Loading fine-tuned model from: {model_dir}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Please train the model first or specify the correct model directory."
        )

    offload_dir = os.path.join(tempfile.gettempdir(), "model_offload")
    os.makedirs(offload_dir, exist_ok=True)

    model = None
    loading_errors = []

    model_arch = getattr(preset, "model_arch", "causal_lm")

    if model_arch == "image_text_to_text":
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        tok = _text_tokenizer(processor)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        print("Model loaded successfully!")
        return model, processor

    try:
        print("Attempting to merge and load model (best for inference - faster generation)...")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if hasattr(peft_model, "merge_and_unload"):
            print("Merging LoRA weights into base model (this may take a minute)...")
            model = peft_model.merge_and_unload()
            print("✓ Model merged and loaded successfully! (Faster inference enabled)")
        else:
            model = peft_model
            print("⚠ Model loaded (merge not available, using PEFT model - will be slower)")
    except Exception as e:
        loading_errors.append(f"Merge strategy failed: {e}")
        print("Merge strategy failed, trying alternative...")

        try:
            print("Attempting to load model without merging...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print("Model loaded successfully (without merging)!")
        except Exception as e2:
            loading_errors.append(f"Direct load failed: {e2}")
            print("Direct load failed, trying sequential device map...")

            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="sequential",
                    torch_dtype=torch.float16,
                )
                print("Model loaded with sequential device map!")
            except Exception as e3:
                loading_errors.append(f"Sequential load failed: {e3}")
                print("Sequential load failed, trying CPU fallback...")

                try:
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        model_dir,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                    )
                    print("Model loaded on CPU (will be slow)!")
                except Exception as e4:
                    loading_errors.append(f"CPU load failed: {e4}")
                    raise RuntimeError(
                        "Failed to load model with all strategies:\n" + "\n".join(loading_errors)
                    )

    if hasattr(model, "config"):
        model.config.use_cache = True

    try:
        if hasattr(torch, "compile") and torch.cuda.is_available():
            print("Compiling model for faster inference (PyTorch 2.0+)...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✓ Model compiled successfully!")
    except Exception as e:
        print(f"Note: Model compilation not available or failed: {e}")
        print("  (This is optional - model will still work, just slightly slower)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=False,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Warning: Error loading tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )

    tok = _text_tokenizer(tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def correct_name_in_text(text, correct_name, similarity_threshold=0.7):
    """Correct misspelled player names in generated text."""
    if not text or not correct_name:
        return text

    name_parts = correct_name.strip().split()
    if not name_parts:
        return text

    corrected_text = text

    name_pattern = re.escape(correct_name)
    corrected_text = re.sub(
        re.compile(name_pattern, re.IGNORECASE),
        correct_name,
        corrected_text,
    )

    if len(name_parts) > 1:
        words = corrected_text.split()
        name_length = len(name_parts)
        replacements = []
        for i in range(len(words) - name_length + 1):
            candidate = " ".join(words[i : i + name_length])
            if candidate.lower() == correct_name.lower():
                continue
            similarity = SequenceMatcher(None, candidate.lower(), correct_name.lower()).ratio()
            if similarity >= similarity_threshold:
                replacements.append((candidate, i))

        if replacements:
            for candidate, _ in reversed(replacements):
                pattern = r"\b" + re.escape(candidate) + r"\b"
                corrected_text = re.sub(pattern, correct_name, corrected_text, flags=re.IGNORECASE)

    if len(name_parts) == 1:
        words = corrected_text.split()
        for word in words:
            if len(word) < 3 or word.lower() == correct_name.lower():
                continue
            common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            if word.lower() in common_words:
                continue
            similarity = SequenceMatcher(None, word.lower(), correct_name.lower()).ratio()
            if similarity >= similarity_threshold:
                pattern = r"\b" + re.escape(word) + r"\b"
                corrected_text = re.sub(pattern, correct_name, corrected_text, flags=re.IGNORECASE)

    return corrected_text


def generate_player_summary(
    name,
    team,
    position,
    top_stats,
    model,
    tokenizer,
    preset,
    max_length=150,
    return_timing=False,
):
    """Generate a summary for a player (training and inference must use the same `preset`)."""
    prompt = player_fields_prompt(name, team, position, top_stats)
    inputs, input_length = preset.build_generation_inputs(tokenizer, prompt)

    device = None
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "base_model") and hasattr(model.base_model, "device"):
        device = model.base_model.device
    elif (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "device")
    ):
        device = model.base_model.model.device
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()

    generation_start_time = time.time()

    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=10,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=_text_tokenizer(tokenizer).pad_token_id
                if _text_tokenizer(tokenizer).pad_token_id is not None
                else _text_tokenizer(tokenizer).eos_token_id,
                eos_token_id=_text_tokenizer(tokenizer).eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        except Exception as e:
            print(f"Warning: Generation error: {e}")
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=_text_tokenizer(tokenizer).pad_token_id
                if _text_tokenizer(tokenizer).pad_token_id is not None
                else _text_tokenizer(tokenizer).eos_token_id,
                eos_token_id=_text_tokenizer(tokenizer).eos_token_id,
            )

    generation_time = time.time() - generation_start_time

    generated_tokens = outputs[0][input_length:]
    generated_summary = _text_tokenizer(tokenizer).decode(generated_tokens, skip_special_tokens=True)
    generated_summary = generated_summary.strip()

    if generated_summary:
        for ending in _BASE_STOP_MARKERS + tuple(preset.extra_generation_stop_markers()):
            if ending in generated_summary:
                generated_summary = generated_summary.split(ending)[0].strip()

    generated_summary = correct_name_in_text(generated_summary, name)

    if return_timing:
        return generated_summary, generation_time
    return generated_summary
