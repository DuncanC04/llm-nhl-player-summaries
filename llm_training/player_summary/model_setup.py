"""Tokenizer, quantized base model, and LoRA."""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _text_tokenizer(tokenizer_or_processor):
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def load_tokenizer(model_name, preset=None):
    """Load tokenizer/processor only (for dataset formatting before full model load)."""
    model_arch = getattr(preset, "model_arch", "causal_lm")
    if model_arch == "image_text_to_text":
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tok = _text_tokenizer(processor)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        return processor

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        print(f"Warning: Error loading tokenizer with fast=False: {e}")
        print("Trying with use_fast=True and additional parameters...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            legacy=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def setup_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.1", use_4bit=True, tokenizer=None, preset=None):
    """Setup model and tokenizer with quantization."""
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes on first run...")
    model_arch = getattr(preset, "model_arch", "causal_lm")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    if tokenizer is None:
        tokenizer = load_tokenizer(model_name, preset=preset)

    trust_remote_code = getattr(preset, "trust_remote_code", True)
    model_cls = AutoModelForImageTextToText if model_arch == "image_text_to_text" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    print("Model loaded successfully!")
    if hasattr(model, "device"):
        print(f"Model device: {model.device}")

    return model, tokenizer


def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """Configure LoRA for efficient fine-tuning."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    return model, peft_config


def print_trainable_parameters(model):
    """Print trainable parameters information."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\ntrainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )
