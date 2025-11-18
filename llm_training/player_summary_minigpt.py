#!/usr/bin/env python3
"""
Player Summary Generator - MiniGPT Model

A lightweight transformer model for generating player summaries.
Based on Mini-GPT architecture - simpler and faster than large language models.
Can run on CPU or basic GPU.

This model uses TensorFlow/Keras and is much smaller (~5M parameters) than
the advanced Mistral-7B model.
"""

import os
import json
import argparse
import string
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

# Use Keras 3 for MiniGPT (transformers compatibility handled in advanced model)
# The advanced model sets TF_USE_LEGACY_KERAS=1, but this model needs Keras 3
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization


# Set environment variables
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """Mask the upper half of the dot product matrix in self attention."""
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def format_player_text(player):
    """Convert player data to text format for training - matches advanced model format"""
    # Format stats like the advanced model
    stats_parts = []
    for stat in player['topStats']:
        stats_parts.append(f"{stat['stat']}: {stat['value']} (percentile: {stat['pctl']})")
    stats_text = "; ".join(stats_parts)
    
    # Use the same clean format as the advanced model
    # This format helps the model learn to generate summaries from input, not memorize structure
    text = (f"Generate a concise player summary based on the following information:\n\n"
            f"Name: {player['name']}\n"
            f"Team: {player['team']}\n"
            f"Position: {player['position']}\n"
            f"Top Statistics: {stats_text}\n\n"
            f"Summary: {player['summary']}")
    
    return text


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
    
    player_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            player_data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(player_data)} player records")
    return player_data


def create_model(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim):
    """Create the MiniGPT model."""
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile("adam", loss=[loss_fn, None])
    return model


class TextGenerator(keras.callbacks.Callback):
    """Generate text after each epoch to see progress"""
    def __init__(self, max_tokens, start_tokens, index_to_word, maxlen, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.maxlen = maxlen  # Model's input sequence length
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        original_start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        current_tokens = [_ for _ in self.start_tokens]
        
        while num_tokens_generated < self.max_tokens:
            # Ensure input is exactly maxlen tokens
            current_length = len(current_tokens)
            
            if current_length > self.maxlen:
                # Truncate to maxlen, keeping the most recent tokens (sliding window)
                x = current_tokens[-self.maxlen:]
                # Sample from the last position in the window
                sample_index = self.maxlen - 1
            elif current_length < self.maxlen:
                # Pad to maxlen with zeros (0 is typically the padding token)
                x = current_tokens + [0] * (self.maxlen - current_length)
                # Sample from the last real token position
                sample_index = current_length - 1
            else:
                # Exactly maxlen tokens
                x = current_tokens
                sample_index = self.maxlen - 1
            
            # Ensure x is exactly maxlen
            assert len(x) == self.maxlen, f"Input length {len(x)} does not match maxlen {self.maxlen}"
            
            x = np.array([x], dtype=np.int32)
            try:
                y, _ = self.model.predict(x, verbose=0)
                sample_token = self.sample_from(y[0][sample_index])
                tokens_generated.append(sample_token)
                current_tokens.append(sample_token)
                num_tokens_generated += 1
                
                # Stop if we hit the end token or invalid token
                if sample_token >= len(self.index_to_word) or sample_token < 0:
                    break
                if self.index_to_word[sample_token] == ']' or self.index_to_word[sample_token] == '<unk>':
                    break
            except Exception as e:
                print(f"Error during generation: {e}")
                break
        
        # Extract only the summary portion (after "Summary:")
        summary_tokens = []
        found_summary = False
        for token in tokens_generated:
            if token < len(self.index_to_word):
                word = self.index_to_word[token]
                # Look for "summary" token (followed by ":")
                if word.lower() == "summary":
                    found_summary = True
                    continue
                if found_summary:
                    # Skip the colon after "summary"
                    if word == ':':
                        continue
                    # All tokens after "summary:" are part of the summary
                    if word and word not in ['<unk>', ':']:
                        summary_tokens.append(token)
        
        # Display only the summary
        if summary_tokens:
            summary_text = " ".join([self.detokenize(_) for _ in summary_tokens if _ < len(self.index_to_word)])
            summary_text = summary_text.replace(" ,", ",").replace(" .", ".").replace("( ", "(").replace(" )", ")").strip()
        else:
            # Fallback: try to extract from full sequence
            full_sequence = original_start_tokens + tokens_generated
            full_words = [self.detokenize(_) for _ in full_sequence if _ < len(self.index_to_word)]
            # Find "summary" and extract after it
            summary_start = False
            summary_words = []
            for word in full_words:
                if word.lower() == "summary":
                    summary_start = True
                    continue
                if summary_start and word != ':' and word not in ['<unk>']:
                    summary_words.append(word)
            summary_text = " ".join(summary_words) if summary_words else " ".join([w for w in full_words if w not in ['<unk>']])
        
        print(f"\n{'='*80}")
        print(f"Generated Summary after epoch {epoch + 1}:")
        print(f"{'='*80}")
        print(summary_text)
        print(f"{'='*80}\n")


def tokenize_text(text, word_to_index, custom_standardization=None):
    """Tokenize text using the same method as training"""
    import string
    import re
    # Apply standardization (lowercase, handle newlines, and punctuation spacing)
    # Replace newlines with spaces (matching training standardization)
    text_processed = re.sub(r'\n+', ' ', text)
    text_lower = text_processed.lower()
    # Add spaces around punctuation to match training tokenization
    for punct in string.punctuation:
        if punct in text_lower:
            text_lower = text_lower.replace(punct, f" {punct} ")
    # Split and clean up multiple spaces
    words = [w for w in text_lower.split() if w]
    # Tokenize - use index 1 for unknown words (OOV token, typically <unk>)
    tokens = [word_to_index.get(word, 1) for word in words]
    return tokens, words


def generate_player_summary(model, word_to_index, vocab, maxlen, name, team, position, stats_list, num_tokens=80, top_k=10, temperature=0.8, return_timing=False):
    """Generate a summary for a player given their stats - returns ONLY the summary text
    
    Args:
        model: The trained model
        word_to_index: Dictionary mapping words to token indices
        vocab: Vocabulary list
        maxlen: Maximum sequence length
        name: Player name
        team: Team abbreviation
        position: Player position
        stats_list: List of (stat_name, value, percentile) tuples
        num_tokens: Maximum number of tokens to generate
        top_k: Top-k sampling parameter
        temperature: Temperature for sampling
        return_timing: If True, return (summary, generation_time) tuple
    
    Returns:
        Generated summary string, or (summary, generation_time) if return_timing=True
    """
    # Format the stats exactly as in training (matches advanced model format)
    stats_parts = []
    for stat_name, value, percentile in stats_list:
        stats_parts.append(f"{stat_name}: {value} (percentile: {percentile})")
    stats_text = "; ".join(stats_parts)
    
    # Create the prompt exactly as in training format (matches advanced model)
    prompt = (f"Generate a concise player summary based on the following information:\n\n"
              f"Name: {name}\n"
              f"Team: {team}\n"
              f"Position: {position}\n"
              f"Top Statistics: {stats_text}\n\n"
              f"Summary:")
    
    # Tokenize using the same method as training
    start_tokens, prompt_words = tokenize_text(prompt, word_to_index)
    prompt_length = len(start_tokens)
    
    # Find where "Summary:" ends - this is where generation starts
    # Look for "summary" followed by ":" to mark the start of generation
    summary_start_token_idx = prompt_length  # Default: after all prompt tokens
    for i, word in enumerate(prompt_words):
        if word.lower() == "summary":
            # The next token should be ":" then we start generating
            summary_start_token_idx = i + 2 if i + 1 < len(prompt_words) and prompt_words[i + 1] == ":" else i + 1
            break
    
    # Generate tokens
    tokens_generated = []
    current_tokens = start_tokens.copy()
    summary_tokens = []  # Track only summary tokens (after "Summary:")
    found_summary_start = False  # Track when we've started generating the summary
    
    # Start timing
    generation_start_time = time.time()
    
    for step in range(num_tokens):
        current_length = len(current_tokens)
        
        # Smart context management: try to preserve prompt while allowing summary generation
        if current_length > maxlen:
            # Need to truncate - try to keep as much prompt as possible
            # Keep prompt + recent summary tokens
            if prompt_length < maxlen:
                # Keep full prompt and recent summary
                keep_prompt = prompt_length
                keep_summary = maxlen - keep_prompt - 1  # -1 for generation position
                if len(tokens_generated) > keep_summary:
                    # Keep recent summary tokens
                    recent_summary = tokens_generated[-keep_summary:]
                    x = start_tokens + recent_summary
                else:
                    # Still have room, pad
                    x = current_tokens[:maxlen]
            else:
                # Prompt is too long, truncate prompt but keep player info (first ~30 tokens)
                # This is a fallback - ideally prompt should fit
                x = current_tokens[-maxlen:]
            sample_index = len(x) - 1
        elif current_length < maxlen:
            # Pad to maxlen with zeros (padding token)
            x = current_tokens + [0] * (maxlen - current_length)
            sample_index = current_length - 1
        else:
            # Exactly maxlen tokens
            x = current_tokens
            sample_index = maxlen - 1
        
        # Ensure x is exactly maxlen
        assert len(x) == maxlen, f"Input length {len(x)} does not match maxlen {maxlen}"
        
        x = np.array([x], dtype=np.int32)
        y, _ = model.predict(x, verbose=0, batch_size=1)
        
        # Sample next token with temperature
        logits = y[0][sample_index]
        
        # Apply temperature for more diverse sampling
        if temperature > 0:
            logits = logits / temperature
            # Filter out invalid tokens
            valid_indices = np.arange(len(logits))
            valid_logits = logits[valid_indices]
            # Top-k sampling
            if top_k > 0:
                top_k = min(top_k, len(valid_logits))
                top_k_indices = np.argsort(valid_logits)[-top_k:]
                top_k_logits = valid_logits[top_k_indices]
                # Softmax
                exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                probs = exp_logits / np.sum(exp_logits)
                # Sample
                sampled_idx = np.random.choice(len(top_k_indices), p=probs)
                next_token = top_k_indices[sampled_idx]
            else:
                # Full sampling
                exp_logits = np.exp(valid_logits - np.max(valid_logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token = np.random.choice(len(valid_indices), p=probs)
        else:
            # Greedy
            next_token = np.argmax(logits)
        
        # Validate token
        if next_token >= len(vocab) or next_token < 0:
            break
        
        # Check for end marker
        token_word = vocab[next_token] if next_token < len(vocab) else "<unk>"
        
        # Track that we're now generating summary (after the prompt)
        if not found_summary_start:
            found_summary_start = True
        
        # In the new format, we don't have a ']' end marker
        # Stop on repeated newlines or other natural endings, or after max tokens
        # For now, we'll generate until num_tokens or natural stopping point
        
        # Skip unknown or empty tokens, but still add them to continue generation
        if token_word == '<unk>' or token_word == '':
            tokens_generated.append(next_token)
            current_tokens.append(next_token)
            # Don't add <unk> to summary tokens
            continue
        
        tokens_generated.append(next_token)
        current_tokens.append(next_token)
        
        # Add to summary tokens (all generated tokens after "Summary:" are summary)
        # Skip the "summary" and ":" tokens themselves
        if token_word.lower() != "summary" and token_word != ':':
            summary_tokens.append(next_token)
        
        # Optional: Stop early if we hit multiple newlines (end of summary)
        # This is a simple heuristic - you might want to refine this
        if len(summary_tokens) > 10:  # Only check after generating some tokens
            # Check last few tokens for natural ending patterns
            recent_words = [vocab[t] if t < len(vocab) else "" for t in summary_tokens[-5:]]
            # Could add logic here to detect natural sentence endings
    
    # End timing
    generation_end_time = time.time()
    generation_time = generation_end_time - generation_start_time
    
    # Extract only the summary portion (everything after "Summary:")
    prompt_structure_words = {"generate", "concise", "player", "summary", "based", "following", 
                             "information", "name", "team", "position", "top", "statistics",
                             "on", "the", "following"}
    
    if summary_tokens:
        # Convert summary tokens to text
        summary_words = [vocab[token] if token < len(vocab) else "" for token in summary_tokens]
        # Filter out empty strings and special formatting tokens
        summary_words = [w for w in summary_words if w and w not in ['<unk>', 'summary', ':'] and not w.startswith('<')]
    else:
        # Fallback: try to extract summary from all generated tokens
        all_words = [vocab[token] if token < len(vocab) else "" for token in tokens_generated]
        # Find "summary" and extract everything after
        summary_start = False
        summary_words = []
        for word in all_words:
            if word.lower() == "summary":
                summary_start = True
                continue
            if summary_start:
                # Skip the colon after "summary"
                if word == ':':
                    continue
                if word and word not in ['<unk>', ':'] and not word.startswith('<'):
                    summary_words.append(word)
    
    # Find where the actual summary content starts (skip prompt repetition)
    # Summaries typically start with descriptive words, player names, or verbs
    # Prompt repetition has structure words like "based", "name", "team", etc.
    summary_start_idx = 0
    prompt_patterns = [
        ["based", "on", "the", "following"],
        ["name", "team", "position"],
        ["top", "statistics"],
    ]
    
    # Look for prompt repetition patterns and skip them
    i = 0
    while i < len(summary_words) - 3:
        word_lower = summary_words[i].lower()
        
        # Check for "based on the following information" pattern
        if word_lower == "based" and i + 3 < len(summary_words):
            if (summary_words[i+1].lower() == "on" and 
                summary_words[i+2].lower() == "the" and 
                summary_words[i+3].lower() == "following"):
                # Skip this entire pattern and find where summary actually starts
                j = i + 4
                # Skip until we find content that doesn't look like prompt structure
                while j < len(summary_words):
                    w = summary_words[j].lower()
                    # Stop when we find a word that's not a prompt structure word
                    if (w not in prompt_structure_words and 
                        w not in [":", "(", ")", ";", "percentile"] and
                        len(w) > 2 and not w.isdigit()):
                        summary_start_idx = j
                        break
                    j += 1
                if summary_start_idx > 0:
                    break
        
        # Check for "name X team Y position Z" pattern
        if word_lower == "name" and i + 4 < len(summary_words):
            # Look ahead to see if this matches input structure
            next_words = [summary_words[i+k].lower() for k in range(1, min(10, len(summary_words)-i))]
            if "team" in next_words[:6] and "position" in next_words[:8]:
                # This is likely prompt repetition
                # Skip until we find actual summary content
                j = i + 1
                while j < len(summary_words):
                    w = summary_words[j].lower()
                    # Look for descriptive words that indicate actual summary
                    if (w not in prompt_structure_words and
                        w not in [":", "(", ")", ";", "percentile", "points", "assists"] and
                        len(w) > 2 and not w.isdigit() and
                        # Check if it's a descriptive word (common summary starters)
                        w not in name.lower().split() and
                        w not in team.lower().split() and
                        w not in position.lower().split()):
                        summary_start_idx = j
                        break
                    j += 1
                if summary_start_idx > 0:
                    break
        
        i += 1
    
    # Extract from the detected start point
    if summary_start_idx > 0:
        cleaned_words = summary_words[summary_start_idx:]
    else:
        # Try a simpler approach: skip words that are clearly prompt structure
        cleaned_words = []
        skip_count = 0
        for i, word in enumerate(summary_words):
            w_lower = word.lower()
            # Skip prompt structure words at the beginning
            if i < 15 and w_lower in prompt_structure_words:
                continue
            # Skip if we see "name:" followed by input values in sequence
            if i > 0 and summary_words[i-1].lower() in ["name", "team", "position"] and w_lower in [name.lower(), team.lower(), position.lower()]:
                continue
            cleaned_words.append(word)
    
    # If we still have prompt-like content, try to find the first descriptive word
    if cleaned_words and cleaned_words[0].lower() in prompt_structure_words:
        for i, word in enumerate(cleaned_words):
            w_lower = word.lower()
            if (w_lower not in prompt_structure_words and
                w_lower not in [":", "(", ")", ";"] and
                len(w_lower) > 2):
                cleaned_words = cleaned_words[i:]
                break
    
    # Join and clean up
    generated_text = " ".join(cleaned_words) if cleaned_words else " ".join(summary_words[-30:])  # Take last 30 words as fallback
    
    # Additional cleaning: Look for the actual summary start by finding descriptive content
    # Summaries often start with player names (capitalized) or descriptive words
    words = generated_text.split()
    if words:
        # Common summary-starting patterns: capitalized words (player names), descriptive adjectives
        summary_start_markers = []
        for i, word in enumerate(words):
            w_clean = word.strip(".,;:()[]{}")
            w_lower = w_clean.lower()
            
            # Skip if it's clearly prompt or structure
            if w_lower in prompt_structure_words or w_lower in [":", "(", ")", ";", "percentile"]:
                continue
            
            # Skip numbers and short words
            if w_clean.isdigit() or len(w_clean) <= 2:
                continue
            
            # Skip if it matches input exactly (unless it's part of a sentence)
            if w_lower in [name.lower(), team.lower(), position.lower()]:
                # Check context - if surrounded by structure, skip
                if i > 0 and words[i-1].lower() in ["name", "team", "position"]:
                    continue
            
            # Look for indicators of actual summary content:
            # Since tokenization is lowercase, we can't use capitalization
            # Instead, look for:
            # 1. Descriptive adjectives (strong, skilled, effective, etc.)
            # 2. Common summary verbs (is, plays, contributes, etc.)
            # 3. Words that are likely player names (multi-word, not in input, not stats)
            # 4. Common summary starter words
            is_descriptive = w_lower in ["strong", "skilled", "effective", "solid", "consistent", "reliable", 
                                        "elite", "good", "great", "excellent", "versatile", "talented",
                                        "offensive", "defensive", "physical", "aggressive", "creative"]
            is_verb = w_lower in ["is", "plays", "contributes", "provides", "brings", "offers", "shows",
                                 "drives", "creates", "helps", "supports", "delivers", "executes"]
            is_likely_player_name = (len(w_clean) > 4 and 
                                    w_lower not in [name.lower(), team.lower(), position.lower()] and
                                    w_lower not in prompt_structure_words and
                                    w_lower not in ["points", "assists", "goals", "percentile", "statistics",
                                                   "name", "team", "position", "top", "based", "following"])
            
            if is_descriptive or is_verb or (is_likely_player_name and i > 5):  # Player names usually come after prompt
                summary_start_markers.append(i)
        
        # If we found summary start markers, use the first one
        if summary_start_markers:
            actual_start = summary_start_markers[0]
            generated_text = " ".join(words[actual_start:])
        else:
            # Fallback: remove first 20 words if they contain prompt structure
            prompt_word_count = sum(1 for w in words[:20] if w.lower() in prompt_structure_words)
            if prompt_word_count > 5:
                generated_text = " ".join(words[20:])
    
    # Clean up the text - fix spacing around punctuation
    generated_text = generated_text.replace(" ,", ",").replace(" .", ".").replace(" '", "'")
    generated_text = generated_text.replace("( ", "(").replace(" )", ")")
    generated_text = generated_text.strip()
    
    # Final safety check: if text is very short or still starts with prompt words, 
    # try to find content after stats
    if len(generated_text.split()) < 5 or (generated_text.split() and generated_text.split()[0].lower() in prompt_structure_words):
        # Look for content after statistical terms
        words = generated_text.split()
        stat_terms = ["points", "assists", "goals", "percentile", "statistics"]
        for i, word in enumerate(words):
            if word.lower() not in stat_terms and word.lower() not in prompt_structure_words and len(word) > 2:
                if i > 10:  # Only if we're past the initial prompt/stats section
                    generated_text = " ".join(words[i:])
                    break
    
    if return_timing:
        return generated_text, generation_time
    return generated_text


def train_model(
    data_path=None,
    output_dir="./player_summary_minigpt.keras",
    vocab_size=20000,  # Increased from 10000 to reduce [UNK] tokens
    maxlen=128,
    embed_dim=256,
    num_heads=4,
    feed_forward_dim=256,
    batch_size=32,
    epochs=50,
    train_split=0.9,
):
    """Train the MiniGPT model."""
    print("=" * 80)
    print("MiniGPT Player Summary Generator - Training")
    print("=" * 80)
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Max Length: {maxlen}")
    print(f"  Embedding Dim: {embed_dim}")
    print(f"  Attention Heads: {num_heads}")
    print(f"  Feed Forward Dim: {feed_forward_dim}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    # Load data
    player_data = load_data(data_path)
    
    # Format data
    print("\nFormatting training data...")
    training_texts = [format_player_text(p) for p in player_data]
    random.shuffle(training_texts)
    
    # Split train/val
    split_idx = int(train_split * len(training_texts))
    train_texts = training_texts[:split_idx]
    val_texts = training_texts[split_idx:]
    
    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    
    # Show a sample of the training format
    if train_texts:
        print("\nSample training text format:")
        print("=" * 80)
        print(train_texts[0][:500] + "..." if len(train_texts[0]) > 500 else train_texts[0])
        print("=" * 80)
    
    # Create dataset
    print("\nPreparing dataset...")
    text_ds = tf_data.Dataset.from_tensor_slices(train_texts)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    
    def custom_standardization(input_string):
        """Lowercase and handle punctuation and newlines"""
        # Replace newlines with spaces (they'll be tokenized as word boundaries)
        no_newlines = tf_strings.regex_replace(input_string, r"\n+", " ")
        lowercased = tf_strings.lower(no_newlines)
        # Add spaces around punctuation
        return tf_strings.regex_replace(lowercased, f"([{string.punctuation}])", r" \1")
    
    # Create vectorization layer
    # Increase vocabulary size to better capture all words from training data
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    
    # Adapt to dataset - this builds vocabulary from training data
    print("Building vocabulary from training data...")
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Check for common structural words in the new format
    # Key words that should be in vocabulary for the new prompt format
    structural_parts = [
        "generate", "concise", "player", "summary", "based", "following", 
        "information", "name", "team", "position", "top", "statistics", 
        "summary", "percentile", ":", "(", ")"
    ]
    print("\nChecking structural word coverage...")
    missing = [w for w in structural_parts if w not in vocab]
    if missing:
        print(f"  ⚠️  Some structural words missing: {missing}")
        print("  Note: This is normal if words are rare. The model will learn from context.")
    else:
        print("  ✓ Key structural words present in vocabulary")
    
    # Show vocabulary coverage statistics
    print(f"\nVocabulary statistics:")
    print(f"  Total tokens: {len(vocab)}")
    print(f"  First 10: {vocab[:10]}")
    print(f"  Last 10: {vocab[-10:]}")
    
    # Count OOV tokens in sample
    sample_text = train_texts[0] if train_texts else ""
    sample_tokens = sample_text.lower().split()
    oov_count = sum(1 for word in sample_tokens if word not in vocab)
    print(f"  OOV tokens in sample text: {oov_count}/{len(sample_tokens)} ({100*oov_count/len(sample_tokens):.1f}%)")
    
    def prepare_lm_inputs_labels(text):
        """Shift word sequences by 1 position for next-token prediction"""
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y
    
    # Prepare the dataset
    text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
    text_ds = text_ds.prefetch(tf_data.AUTOTUNE)
    
    # Create model
    print("\nCreating model...")
    model = create_model(maxlen, vocab_size, embed_dim, num_heads, feed_forward_dim)
    model.summary()
    
    # Set up text generation callback
    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index
    
    # Use the same format as training for the callback prompt
    start_prompt = ("Generate a concise player summary based on the following information:\n\n"
                    "Name: Connor McDavid\n"
                    "Team: EDM\n"
                    "Position: C\n"
                    "Top Statistics: points: 132 (percentile: 99); assists: 89 (percentile: 99)\n\n"
                    "Summary:")
    # Tokenize the prompt using the same method
    start_tokens, _ = tokenize_text(start_prompt, word_to_index)
    text_gen_callback = TextGenerator(50, start_tokens, vocab, maxlen, print_every=5)
    
    # Train
    print("\nStarting training...")
    print(f"Training for {epochs} epochs")
    print("Sample text will be generated every 5 epochs\n")
    
    steps_per_epoch = len(train_texts) // batch_size
    
    # Start timing
    training_start_time = time.time()
    
    history = model.fit(
        text_ds,
        verbose=1,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[text_gen_callback]
    )
    
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    # Save model and tokenizer info
    print(f"\nSaving model to: {output_dir}")
    model.save(output_dir)
    
    # Save vocabulary for later use
    vocab_file = str(output_dir).replace('.keras', '_vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    
    print(f"Vocabulary saved to: {vocab_file}")
    
    # Print training time metrics
    print("\n" + "="*80)
    print("TRAINING TIME METRICS")
    print("="*80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Time per epoch: {training_time/epochs:.2f} seconds ({training_time/epochs/60:.2f} minutes)")
    print("="*80)
    print("\nTraining complete!")
    
    return model, vocab, word_to_index


def test_model(model_path, data_path=None, num_test=3):
    """Test the trained model on sample data."""
    print("=" * 80)
    print("Testing MiniGPT Model")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path, custom_objects={
        'TransformerBlock': TransformerBlock,
        'TokenAndPositionEmbedding': TokenAndPositionEmbedding
    })
    
    # Load vocabulary
    vocab_file = str(model_path).replace('.keras', '_vocab.json')
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index
    
    # Get maxlen from model input shape
    maxlen = model.input_shape[1]
    
    # Load test data
    player_data = load_data(data_path)
    
    print(f"\nTesting on {num_test} examples:\n")
    
    # Track timing for all generations
    generation_times = []
    
    for i in range(min(num_test, len(player_data))):
        player = player_data[i]
        stats_list = [(s['stat'], s['value'], s['pctl']) for s in player['topStats']]
        
        generated, gen_time = generate_player_summary(
            model, word_to_index, vocab, maxlen,
            player['name'],
            player['team'],
            player['position'],
            stats_list,
            return_timing=True
        )
        generation_times.append(gen_time)
        
        print(f"{'='*80}")
        print(f"Player: {player['name']} ({player['team']} - {player['position']})")
        print(f"{'='*80}")
        print(f"\nORIGINAL SUMMARY:")
        print(player['summary'])
        print(f"\nGENERATED SUMMARY:")
        print(generated)
        print(f"\n[Generation time: {gen_time:.3f} seconds]")
        print(f"{'='*80}\n")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Train or test MiniGPT model for player summaries"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL data file (default: auto-detect)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./player_summary_minigpt.keras",
        help="Output path for trained model (default: ./player_summary_minigpt.keras)"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test an existing model (skip training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="Vocabulary size (default: 20000, increased to reduce [UNK] tokens)"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    
    args = parser.parse_args()
    
    # Check TensorFlow/Keras
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
    
    if args.test_only:
        test_model(args.output_dir, args.data_path)
    else:
        train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            maxlen=args.maxlen,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()

