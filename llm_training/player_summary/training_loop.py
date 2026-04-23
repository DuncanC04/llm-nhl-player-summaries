"""SFTTrainer wiring and checkpoint save."""

import time

import torch
from trl import SFTConfig, SFTTrainer


def _text_tokenizer(tokenizer_or_processor):
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir="./results",
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_seq_length=512,
    eval_steps=50,
    save_steps=50,
):
    """Train the model (expects `model` to already be a PEFT-wrapped causal LM)."""
    # FP16 + GradScaler can break when grads are BF16 (common with 4-bit + compute). Use BF16 on CUDA Ampere+.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
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
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=3,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
    )

    print("\nTraining configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Max sequence length: {training_args.max_length}")

    # Model is already PeftModel from setup_lora; TRL 1.x forbids passing peft_config in that case.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=_text_tokenizer(tokenizer),
        args=training_args,
    )

    print("\nTrainer initialized successfully!")
    print("\nStarting training...")
    print("You can monitor GPU usage in Task Manager or with nvidia-smi\n")

    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    print("\n" + "=" * 80)
    print("TRAINING TIME METRICS")
    print("=" * 80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(
        f"Time per epoch: {training_time/num_epochs:.2f} seconds ({training_time/num_epochs/60:.2f} minutes)"
    )
    print("=" * 80)

    return trainer


def save_model(trainer, tokenizer, output_dir="./player_summary_model"):
    """Save the fine-tuned model."""
    print(f"\nSaving model to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")
