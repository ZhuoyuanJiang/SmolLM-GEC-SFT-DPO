#!/usr/bin/env python3
"""
SFT Training Script for SmolLM GEC
Supports both padding and packing approaches
"""

import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json

# Import our utility functions
from utils import (
    set_seed, setup_tokenizer_once, format_dataset, 
    evaluate_model, save_results, create_output_dir, get_gpu_memory_usage
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for SmolLM GEC")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="Base model to fine-tune")
    
    # Training method
    parser.add_argument("--method", type=str, required=True, choices=["padding", "packing"],
                        help="Training method: padding or packing")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Dataset arguments
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--dataset_name", type=str, default="grammarly/coedit",
                        help="Dataset to use")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--base_output_dir", type=str, default="experiments",
                        help="Base directory for all experiments")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    
    return parser.parse_args()


def load_and_prepare_data(args):
    """Load and prepare the GEC dataset"""
    print(f"Loading dataset: {args.dataset_name}")
    
    # Load full dataset
    full_train_ds = load_dataset(args.dataset_name, split="train")
    full_test_ds = load_dataset(args.dataset_name, split="validation")
    
    # Filter for GEC task only
    train_ds = full_train_ds.filter(lambda example: example['task'] == 'gec')
    test_ds = full_test_ds.filter(lambda example: example['task'] == 'gec')
    
    print(f"GEC Train dataset size: {len(train_ds)}")
    print(f"GEC Test dataset size: {len(test_ds)}")
    
    return train_ds, test_ds


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    if args.output_dir is None:
        # Auto-generate output directory name
        exp_name = f"sft_{args.method}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.num_epochs}"
        args.output_dir = create_output_dir(args.base_output_dir, exp_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False if args.gradient_checkpointing else True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup tokenizer with proper pad token and [SEP] token
    model, tokenizer = setup_tokenizer_once(model, tokenizer)
    
    # Load and prepare datasets
    train_ds, test_ds = load_and_prepare_data(args)
    
    # Format datasets
    print("Formatting datasets...")
    formatted_train_ds = format_dataset(train_ds, tokenizer)
    formatted_test_ds = format_dataset(test_ds, tokenizer)
    
    # Configure SFT training
    training_args = SFTConfig(
        # Output & Logging
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        
        # Training parameters
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        
        # Batch settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Mixed precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        
        # Memory optimizations
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        
        # Dataset configuration
        max_seq_length=args.max_seq_length,
        packing=(args.method == "packing"),
        dataset_text_field="text",
        remove_unused_columns=True,
        
        # Saving strategy
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        
        # Evaluation strategy
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none" if not args.push_to_hub else "tensorboard",
        
        # Hub settings
        push_to_hub=args.push_to_hub,
        hub_model_id=f"smollm-gec-{args.method}" if args.push_to_hub else None,
    )
    
    # Create data collator for padding approach
    data_collator = None
    if args.method == "padding":
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="[SEP]",
            tokenizer=tokenizer,
            mlm=False
        )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_ds,
        eval_dataset=formatted_test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Record start time
    start_time = time.time()
    
    # Log initial GPU memory
    print(f"Before training: {get_gpu_memory_usage()}")
    
    # Train the model
    print(f"Starting SFT training with {args.method} method...")
    train_result = trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save the model
    print("Saving model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Evaluate on test set
    print("Evaluating on test set...")
    bleu_score = evaluate_model(model, tokenizer, test_ds, show_progress=True)
    print(f"Test BLEU Score: {bleu_score:.4f}")
    
    # Get final metrics
    final_train_loss = train_result.metrics.get("train_loss", 0.0)
    
    # Try to get eval loss from trainer state
    eval_loss = None
    if hasattr(trainer.state, 'log_history'):
        for entry in reversed(trainer.state.log_history):
            if 'eval_loss' in entry:
                eval_loss = entry['eval_loss']
                break
    
    # Save results
    config = {
        "method": args.method,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "model_name": args.model_name,
        "gradient_checkpointing": args.gradient_checkpointing
    }
    
    metrics = {
        "bleu_score": bleu_score,
        "train_loss": final_train_loss,
        "eval_loss": eval_loss,
        "training_time": training_time,
        "train_samples": len(formatted_train_ds),
        "test_samples": len(test_ds)
    }
    
    save_results(args.output_dir, config, metrics)
    
    # Final memory usage
    print(f"After training: {get_gpu_memory_usage()}")
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Method: {args.method}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Output Directory: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()