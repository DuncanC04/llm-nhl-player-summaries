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
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
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
    """Convert player data to text format for training"""
    stats_parts = []
    for stat in player['topStats']:
        stats_parts.append(f"{stat['stat']} {stat['value']} percentile {stat['pctl']}")
    
    stats_text = " , ".join(stats_parts)
    
    text = (f"[PLAYER: {player['name']} | TEAM: {player['team']} | "
            f"POS: {player['position']} | STATS: {stats_text} | "
            f"SUMMARY: {player['summary']}]")
    
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
    def __init__(self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
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
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        maxlen = len(self.start_tokens) + self.max_tokens
        
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
            
            if self.index_to_word[sample_token] == ']':
                break
        
        txt = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"\n{'='*80}")
        print(f"Generated text after epoch {epoch + 1}:")
        print(f"{'='*80}")
        print(txt)
        print(f"{'='*80}\n")


def generate_player_summary(model, word_to_index, vocab, maxlen, name, team, position, stats_list, num_tokens=60, top_k=5):
    """Generate a summary for a player given their stats"""
    # Format the stats
    stats_parts = []
    for stat_name, value, percentile in stats_list:
        stats_parts.append(f"{stat_name} {value} percentile {percentile}")
    stats_text = " , ".join(stats_parts)
    
    # Create the prompt
    prompt = f"[PLAYER: {name} | TEAM: {team} | POS: {position} | STATS: {stats_text} | SUMMARY:"
    
    # Tokenize
    start_tokens = [word_to_index.get(word, 1) for word in prompt.lower().split()]
    
    # Generate
    tokens_generated = []
    current_tokens = start_tokens.copy()
    
    for _ in range(num_tokens):
        pad_len = maxlen - len(current_tokens)
        if pad_len < 0:
            x = current_tokens[:maxlen]
            sample_index = maxlen - 1
        elif pad_len > 0:
            x = current_tokens + [0] * pad_len
            sample_index = len(current_tokens) - 1
        else:
            x = current_tokens
            sample_index = maxlen - 1
        
        x = np.array([x])
        y, _ = model.predict(x, verbose=0)
        
        # Sample next token
        logits = y[0][sample_index]
        logits, indices = ops.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        next_token = np.random.choice(indices, p=preds)
        
        # Stop if we hit end marker
        if vocab[next_token] == ']':
            break
        
        tokens_generated.append(next_token)
        current_tokens.append(next_token)
    
    # Convert back to text
    generated_text = " ".join([vocab[token] for token in tokens_generated])
    
    # Clean up the text
    generated_text = generated_text.replace(" ,", ",").replace(" .", ".")
    
    return generated_text


def train_model(
    data_path=None,
    output_dir="./player_summary_minigpt.keras",
    vocab_size=10000,
    maxlen=128,
    embed_dim=256,
    num_heads=4,
    feed_forward_dim=256,
    batch_size=32,
    epochs=20,
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
    
    # Create dataset
    print("\nPreparing dataset...")
    text_ds = tf_data.Dataset.from_tensor_slices(train_texts)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    
    def custom_standardization(input_string):
        """Lowercase and handle punctuation"""
        lowercased = tf_strings.lower(input_string)
        return tf_strings.regex_replace(lowercased, f"([{string.punctuation}])", r" \1")
    
    # Create vectorization layer
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )
    
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()
    
    print(f"Vocabulary size: {len(vocab)}")
    
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
    
    start_prompt = "[PLAYER: Connor McDavid | TEAM: EDM | POS: C | STATS: points 132 percentile 99 , assists 89 percentile 99 | SUMMARY:"
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.lower().split()]
    text_gen_callback = TextGenerator(50, start_tokens, vocab, print_every=5)
    
    # Train
    print("\nStarting training...")
    print(f"Training for {epochs} epochs")
    print("Sample text will be generated every 5 epochs\n")
    
    steps_per_epoch = len(train_texts) // batch_size
    
    history = model.fit(
        text_ds,
        verbose=1,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[text_gen_callback]
    )
    
    # Save model and tokenizer info
    print(f"\nSaving model to: {output_dir}")
    model.save(output_dir)
    
    # Save vocabulary for later use
    vocab_file = str(output_dir).replace('.keras', '_vocab.json')
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    
    print(f"Vocabulary saved to: {vocab_file}")
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
    
    for i in range(min(num_test, len(player_data))):
        player = player_data[i]
        stats_list = [(s['stat'], s['value'], s['pctl']) for s in player['topStats']]
        
        generated = generate_player_summary(
            model, word_to_index, vocab, maxlen,
            player['name'],
            player['team'],
            player['position'],
            stats_list
        )
        
        print(f"{'='*80}")
        print(f"Player: {player['name']} ({player['team']} - {player['position']})")
        print(f"{'='*80}")
        print(f"\nORIGINAL SUMMARY:")
        print(player['summary'])
        print(f"\nGENERATED SUMMARY:")
        print(generated)
        print(f"{'='*80}\n")


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
        default=10000,
        help="Vocabulary size (default: 10000)"
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

