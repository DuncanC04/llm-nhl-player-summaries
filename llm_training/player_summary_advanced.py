#!/usr/bin/env python3
"""
Player Statistics Summary Generator - LLM Fine-tuning

This script fine-tunes a language model to generate player summaries based on 
name, team, position, and top statistics.

Hardware Requirements:
- NVIDIA GPU with at least 8GB VRAM (RTX 3060 or better recommended)
- CUDA-enabled PyTorch
- With 4-bit quantization, 8GB VRAM is sufficient

Model: Uses Mistral-7B with QLoRA for efficient training on consumer hardware.
"""

import json
import os
import argparse
import gc
import time
import re
from pathlib import Path
from difflib import SequenceMatcher

# Set environment variable for Keras compatibility with transformers
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig
import warnings

warnings.filterwarnings('ignore')


def check_cuda():
    """Check CUDA availability and print GPU information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("⚠️  WARNING: CUDA is not available. Training will be very slow on CPU.")
        return False


def find_data_file():
    """Find the data file by trying multiple possible paths."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    possible_paths = [
        script_dir.parent.parent / "Data" / "out" / "aiTop10Stats_complete.jsonl",
        project_root / "Data" / "out" / "aiTop10Stats_complete.jsonl",
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
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples")
    if len(data) > 0:
        print("\nSample entry:")
        print(json.dumps(data[0], indent=2))
    
    return data


def format_stats_text(stats):
    """Convert topStats array to readable text."""
    stats_text = []
    for stat in stats:
        stats_text.append(f"{stat['stat']}: {stat['value']} (percentile: {stat['pctl']})")
    return "; ".join(stats_text)


def create_prompt(example):
    """Create a formatted prompt for training."""
    stats_text = format_stats_text(example['topStats'])
    
    prompt = f"""Generate a concise player summary based on the following information:

Name: {example['name']}
Team: {example['team']}
Position: {example['position']}
Top Statistics: {stats_text}

Summary:"""
    
    return prompt


def create_training_example(example):
    """Create full training text with prompt and completion."""
    prompt = create_prompt(example)
    full_text = f"{prompt} {example['summary']}"
    return {"text": full_text}


def prepare_datasets(data, train_split=0.9):
    """Prepare train and validation datasets."""
    # Create formatted dataset
    formatted_data = [create_training_example(example) for example in data]
    
    print("\nSample formatted training example:")
    print(formatted_data[0]['text'])
    print(f"\nTotal examples: {len(formatted_data)}")
    
    # Convert to pandas DataFrame for easy splitting
    df = pd.DataFrame(formatted_data)
    
    # Split: 90% train, 10% validation
    train_size = int(train_split * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    return dataset_dict, train_size


def setup_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.1", use_4bit=True):
    """Setup model and tokenizer with quantization."""
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes on first run...")
    
    # Quantization config for 4-bit loading (saves memory)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer with error handling for version compatibility
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False  # Use slow tokenizer if fast tokenizer has issues
        )
    except Exception as e:
        print(f"Warning: Error loading tokenizer with fast=False: {e}")
        print("Trying with use_fast=True and additional parameters...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            legacy=False
        )
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    print("Model loaded successfully!")
    if hasattr(model, 'device'):
        print(f"Model device: {model.device}")
    
    return model, tokenizer


def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """Configure LoRA for efficient fine-tuning."""
    # LoRA configuration
    peft_config = LoraConfig(
        r=r,  # LoRA rank
        lora_alpha=lora_alpha,  # LoRA alpha
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral attention modules
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
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


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    peft_config,
    output_dir="./results",
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_seq_length=512,
    eval_steps=50,
    save_steps=50,
):
    """Train the model."""
    # SFTConfig extends TrainingArguments with SFT-specific parameters
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=save_steps,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=3,
        # SFT-specific parameters
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
    )
    
    print("\nTraining configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Max sequence length: {training_args.max_length}")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("\nTrainer initialized successfully!")
    print("\nStarting training...")
    print("You can monitor GPU usage in Task Manager or with nvidia-smi\n")
    
    # Start timing
    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    print("\n" + "="*80)
    print("TRAINING TIME METRICS")
    print("="*80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Time per epoch: {training_time/num_epochs:.2f} seconds ({training_time/num_epochs/60:.2f} minutes)")
    print("="*80)
    
    return trainer


def save_model(trainer, tokenizer, output_dir="./player_summary_model"):
    """Save the fine-tuned model."""
    print(f"\nSaving model to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved successfully!")


def load_finetuned_model(model_dir="./player_summary_model"):
    """Load a fine-tuned model for inference."""
    print(f"Loading fine-tuned model from: {model_dir}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Please train the model first or specify the correct model directory."
        )
    
    # Create a temporary offload directory if needed
    import tempfile
    offload_dir = os.path.join(tempfile.gettempdir(), "model_offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    # Try different loading strategies
    model = None
    loading_errors = []
    
    # Strategy 1: Try merging and loading (recommended for inference - MUCH faster)
    try:
        print("Attempting to merge and load model (best for inference - faster generation)...")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Merge LoRA weights into base model for faster inference
        # This is critical for speed - merged models are 2-3x faster
        if hasattr(peft_model, 'merge_and_unload'):
            print("Merging LoRA weights into base model (this may take a minute)...")
            model = peft_model.merge_and_unload()
            print("✓ Model merged and loaded successfully! (Faster inference enabled)")
        else:
            # If merge_and_unload not available, use the model as-is
            model = peft_model
            print("⚠ Model loaded (merge not available, using PEFT model - will be slower)")
    except Exception as e:
        loading_errors.append(f"Merge strategy failed: {e}")
        print(f"Merge strategy failed, trying alternative...")
        
        # Strategy 2: Load without merging
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
            print(f"Direct load failed, trying sequential device map...")
            
            # Strategy 3: Sequential device map
            try:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="sequential",
                    torch_dtype=torch.float16,
                )
                print("Model loaded with sequential device map!")
            except Exception as e3:
                loading_errors.append(f"Sequential load failed: {e3}")
                print(f"Sequential load failed, trying CPU fallback...")
                
                # Strategy 4: CPU fallback (slow but should work)
                try:
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        model_dir,
                        device_map="cpu",
                        torch_dtype=torch.float32,  # Use float32 for CPU
                    )
                    print("Model loaded on CPU (will be slow)!")
                except Exception as e4:
                    loading_errors.append(f"CPU load failed: {e4}")
                    raise RuntimeError(
                        f"Failed to load model with all strategies:\n" + 
                        "\n".join(loading_errors)
                    )
    
    # Enable cache for inference (was disabled during training) - CRITICAL for speed
    if hasattr(model, 'config'):
        model.config.use_cache = True
    
    # Try to compile model for faster inference (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            print("Compiling model for faster inference (PyTorch 2.0+)...")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("✓ Model compiled successfully!")
    except Exception as e:
        print(f"Note: Model compilation not available or failed: {e}")
        print("  (This is optional - model will still work, just slightly slower)")
    
    # Load tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=False
        )
    except Exception as e:
        print(f"Warning: Error loading tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to eval mode for inference
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def correct_name_in_text(text, correct_name, similarity_threshold=0.7):
    """Correct misspelled player names in generated text.
    
    Args:
        text: The generated text that may contain misspelled names
        correct_name: The correct player name to use for replacement
        similarity_threshold: Minimum similarity ratio (0-1) to consider a match
    
    Returns:
        Text with corrected names
    """
    if not text or not correct_name:
        return text
    
    # Split name into parts (first and last name)
    name_parts = correct_name.strip().split()
    if not name_parts:
        return text
    
    # Create patterns for matching names
    # We'll look for sequences of words that might be the name
    corrected_text = text
    
    # Method 1: Try exact case-insensitive match first (fast path)
    # This handles cases where the name is correct but just has wrong capitalization
    name_pattern = re.escape(correct_name)
    corrected_text = re.sub(
        re.compile(name_pattern, re.IGNORECASE),
        correct_name,
        corrected_text
    )
    
    # Method 2: For multi-word names, try to find similar sequences
    if len(name_parts) > 1:
        # Look for sequences of words that might be the name
        words = corrected_text.split()
        name_length = len(name_parts)
        
        # Check sequences of words that match the length of the name
        # Build replacement map first to avoid modifying text during iteration
        replacements = []
        for i in range(len(words) - name_length + 1):
            candidate = ' '.join(words[i:i + name_length])
            
            # Quick check: skip if already correct (case-insensitive)
            if candidate.lower() == correct_name.lower():
                continue
            
            # Calculate similarity (fast for short strings)
            similarity = SequenceMatcher(None, candidate.lower(), correct_name.lower()).ratio()
            
            # If similarity is high enough, mark for replacement
            if similarity >= similarity_threshold:
                replacements.append((candidate, i))
        
        # Apply replacements (from end to start to preserve indices)
        if replacements:
            # Use a single regex pass for all replacements
            for candidate, _ in reversed(replacements):
                pattern = r'\b' + re.escape(candidate) + r'\b'
                corrected_text = re.sub(pattern, correct_name, corrected_text, flags=re.IGNORECASE)
    
    # Method 3: For single-word names, check each word individually
    # This handles cases where a single-word name is misspelled
    if len(name_parts) == 1:
        # Single name - check each word in the text
        words = corrected_text.split()
        for word in words:
            # Skip if word is too short, already matches, or is a common word
            if len(word) < 3 or word.lower() == correct_name.lower():
                continue
            
            # Skip common words that might accidentally match
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            if word.lower() in common_words:
                continue
            
            similarity = SequenceMatcher(None, word.lower(), correct_name.lower()).ratio()
            if similarity >= similarity_threshold:
                # Replace the word with correct name
                pattern = r'\b' + re.escape(word) + r'\b'
                corrected_text = re.sub(pattern, correct_name, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text


def generate_player_summary(name, team, position, top_stats, model, tokenizer, max_length=150, return_timing=False):
    """Generate a summary for a player.
    
    Args:
        name: Player name
        team: Team abbreviation
        position: Player position
        top_stats: List of top statistics
        model: The model to use for generation
        tokenizer: The tokenizer
        max_length: Maximum length of generated summary
        return_timing: If True, return (summary, generation_time) tuple
    
    Returns:
        Generated summary string, or (summary, generation_time) if return_timing=True
    """
    stats_text = format_stats_text(top_stats)
    
    # Use the same prompt format as training
    prompt = f"""Generate a concise player summary based on the following information:

Name: {name}
Team: {team}
Position: {position}
Top Statistics: {stats_text}

Summary:"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    # Determine device - handle PEFT models correctly
    device = None
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'device'):
        device = model.base_model.device
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'device'):
        device = model.base_model.model.device
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to eval mode for inference
    model.eval()
    
    # Start timing
    generation_start_time = time.time()
    
    # Generate with better parameters
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_new_tokens=10,  # Ensure minimum generation
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Prevent repetition (increased from 1.1)
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            )
        except Exception as e:
            print(f"Warning: Generation error: {e}")
            # Fallback to simpler generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,  # Use greedy decoding as fallback
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    generation_end_time = time.time()
    generation_time = generation_end_time - generation_start_time
    
    # Decode only the newly generated tokens (not the prompt)
    generated_tokens = outputs[0][input_length:]
    generated_summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up the summary
    generated_summary = generated_summary.strip()
    
    # Remove any trailing incomplete sentences or special tokens
    if generated_summary:
        # Stop at common sentence endings or EOS patterns
        for ending in ['\n\n', '\nSummary:', 'Summary:', '###', '---']:
            if ending in generated_summary:
                generated_summary = generated_summary.split(ending)[0].strip()
    
    # Post-process to correct name misspellings
    generated_summary = correct_name_in_text(generated_summary, name)
    
    if return_timing:
        return generated_summary, generation_time
    return generated_summary


def test_model(model, tokenizer, data, train_size, num_test=3):
    """Test the model on validation examples."""
    print("\nTesting model on validation examples:\n")
    print("=" * 80)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Calculate validation set size
    val_set_size = len(data) - train_size
    if num_test > val_set_size:
        num_test = val_set_size
        print(f"Note: Only {val_set_size} examples in validation set, testing on all of them.\n")
    elif num_test == -1:
        num_test = val_set_size
        print(f"Testing on all {val_set_size} validation examples.\n")
    
    # Test on examples from the validation set
    # Spread them evenly across the validation set
    if num_test == 1:
        test_indices = [0]
    else:
        step = max(1, val_set_size // num_test)
        test_indices = list(range(0, val_set_size, step))[:num_test]
    
    print(f"Testing on {len(test_indices)} players from validation set:\n")
    
    # Track timing for all generations
    generation_times = []
    
    for i, idx in enumerate(test_indices, 1):
        if idx < val_set_size:
            example = data[train_size + idx]
            
            print(f"\n[{i}/{len(test_indices)}] Player: {example['name']}")
            print(f"Team: {example['team']} | Position: {example['position']}")
            print(f"Top Stats: {format_stats_text(example['topStats'])}")
            
            print("\n--- ORIGINAL SUMMARY ---")
            print(example['summary'])
            
            print("\n--- GENERATED SUMMARY ---")
            try:
                generated, gen_time = generate_player_summary(
                    example['name'],
                    example['team'],
                    example['position'],
                    example['topStats'],
                    model,
                    tokenizer,
                    return_timing=True
                )
                generation_times.append(gen_time)
                if generated:
                    print(generated)
                    print(f"\n[Generation time: {gen_time:.3f} seconds]")
                else:
                    print("[WARNING: Empty generation - model may need more training]")
            except Exception as e:
                print(f"[ERROR generating summary: {e}]")
            print("=" * 80)
    
    # Print summary statistics
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


def generate_all_summaries(data, model, tokenizer, output_file="generated_summaries.jsonl"):
    """Generate summaries for all players and save to file."""
    results = []
    generation_times = []
    
    total_start_time = time.time()
    
    for i, example in enumerate(data):
        if i % 10 == 0:
            print(f"Processing {i}/{len(data)}...")
        
        generated_summary, gen_time = generate_player_summary(
            example['name'],
            example['team'],
            example['position'],
            example['topStats'],
            model,
            tokenizer,
            return_timing=True
        )
        generation_times.append(gen_time)
        
        result = {
            "name": example['name'],
            "team": example['team'],
            "position": example['position'],
            "topStats": example['topStats'],
            "original_summary": example['summary'],
            "generated_summary": generated_summary
        }
        results.append(result)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nAll summaries generated and saved to: {output_file}")
    
    # Print timing statistics
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


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory cleared!")


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for player summary generation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL data file (default: auto-detect)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model name to fine-tune (default: mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./player_summary_model",
        help="Directory to save the fine-tuned model (default: ./player_summary_model)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Train/validation split ratio (default: 0.9)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test an existing model (skip training)"
    )
    parser.add_argument(
        "--generate_all",
        action="store_true",
        help="Generate summaries for all players after training/testing"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more memory)"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=3,
        help="Number of test examples to generate after training (default: 3, use -1 for all validation examples)"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    cuda_available = check_cuda()
    if not cuda_available:
        print("\n" + "="*80)
        print("ERROR: CUDA is not available!")
        print("="*80)
        print("\nThe advanced model requires a CUDA-enabled GPU.")
        print("\nYour system has an NVIDIA GPU, but PyTorch was installed")
        print("without CUDA support (CPU-only version).")
        print("\nTo fix this:")
        print("  1. Run: install_pytorch_cuda.bat (Windows) or install_pytorch_cuda.sh (Linux/Mac)")
        print("  2. Or manually:")
        print("     python -m pip uninstall -y torch torchvision torchaudio")
        print("     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Restart your terminal/Python environment")
        print("\nAlternative: Use the simple MiniGPT model which works on CPU:")
        print("  python llm_training/player_summary_minigpt.py")
        print("="*80)
        return
    
    if args.test_only:
        # Only test mode
        print("\n" + "="*80)
        print("TEST MODE: Loading fine-tuned model for inference")
        print("="*80)
        print("\nNOTE: Make sure the model has been trained first!")
        print("If you see poor quality output, the model may need more training.\n")
        
        model, tokenizer = load_finetuned_model(args.output_dir)
        data = load_data(args.data_path)
        dataset_dict, train_size = prepare_datasets(data, args.train_split)
        test_model(model, tokenizer, data, train_size, num_test=args.num_test)
        if args.generate_all:
            generate_all_summaries(data, model, tokenizer)
        cleanup_memory()
        return
    
    # Training mode
    # Load data
    data = load_data(args.data_path)
    
    # Prepare datasets
    dataset_dict, train_size = prepare_datasets(data, args.train_split)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_4bit=not args.no_4bit
    )
    
    # Setup LoRA
    model, peft_config = setup_lora(model)
    
    # Train model
    trainer = train_model(
        model,
        tokenizer,
        dataset_dict['train'],
        dataset_dict['validation'],
        peft_config,
        output_dir="./results",
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )
    
    # Save model
    save_model(trainer, tokenizer, args.output_dir)
    
    # Test model
    test_model(model, tokenizer, data, train_size, num_test=args.num_test)
    
    # Generate all summaries if requested
    if args.generate_all:
        generate_all_summaries(data, model, tokenizer)
    
    # Cleanup
    cleanup_memory()
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()

