#!/usr/bin/env python3
"""
DPO/IPO Training Script for SmolLM GEC
Supports both DPO and IPO preference optimization methods
"""

import argparse
import os
import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer

# Import our utility functions
from utils import (
    set_seed, setup_tokenizer_once, evaluate_model, 
    save_results, create_output_dir, get_gpu_memory_usage
)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO/IPO training for SmolLM GEC")
    
    # Model arguments
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to the SFT model to start from")
    parser.add_argument("--base_model_name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="Base model name for reference model")
    
    # Training method
    parser.add_argument("--method", type=str, required=True, choices=["dpo", "ipo"],
                        help="Preference optimization method: dpo or ipo")
    
    # Dataset arguments
    parser.add_argument("--preference_dataset_path", type=str, required=True,
                        help="Path to preference dataset")
    parser.add_argument("--test_dataset_name", type=str, default="grammarly/coedit",
                        help="Test dataset for evaluation")
    
    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="Learning rate for preference optimization")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO/IPO beta parameter")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    
    # Batch settings
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    
    # Sequence length settings
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=128,
                        help="Maximum prompt length")
    
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
    """Load preference dataset and test dataset"""
    print(f"Loading preference dataset from: {args.preference_dataset_path}")
    
    # Load preference dataset
    preference_dataset = load_from_disk(args.preference_dataset_path)
    print(f"Preference dataset size: {len(preference_dataset)}")
    
    # Split into train/validation (90/10 split)
    train_size = int(0.9 * len(preference_dataset))
    train_dataset = preference_dataset.select(range(train_size))
    eval_dataset = preference_dataset.select(range(train_size, len(preference_dataset)))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load test dataset for final evaluation
    from datasets import load_dataset
    full_test_ds = load_dataset(args.test_dataset_name, split="validation")
    test_ds = full_test_ds.filter(lambda example: example['task'] == 'gec')
    print(f"Test dataset size: {len(test_ds)}")
    
    return train_dataset, eval_dataset, test_ds


def format_dataset_for_dpo(dataset):
    """Format dataset for DPO training - ensure correct field names"""
    def format_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
    
    return dataset.map(format_example)


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    if args.output_dir is None:
        # Extract SFT config from path for naming
        sft_path_parts = args.sft_model_path.split('/')[-1]
        exp_name = f"{args.method}_{sft_path_parts}_lr{args.learning_rate}_ep{args.num_epochs}"
        args.output_dir = create_output_dir(args.base_output_dir, exp_name)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    # Load datasets
    train_dataset, eval_dataset, test_ds = load_and_prepare_data(args)
    
    # Format datasets for DPO
    train_dataset = format_dataset_for_dpo(train_dataset)
    eval_dataset = format_dataset_for_dpo(eval_dataset)
    
    # Load SFT model
    print(f"Loading SFT model from: {args.sft_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        use_cache=False if args.gradient_checkpointing else True
    )
    
    # Load reference model (base model)
    print(f"Loading reference model: {args.base_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Load tokenizer (try from SFT model first, fallback to base)
    tokenizer_path = args.sft_model_path
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        print("Warning: Using base model tokenizer")
        tokenizer_path = args.base_model_name
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Setup both models with the same tokenizer configuration
    model, tokenizer = setup_tokenizer_once(model, tokenizer)
    ref_model, _ = setup_tokenizer_once(ref_model, tokenizer)
    
    # Verify model configurations match
    assert model.config.vocab_size == ref_model.config.vocab_size, "Vocab size mismatch between models!"
    assert model.config.vocab_size == len(tokenizer), "Model-tokenizer vocab size mismatch!"
    
    print(f"Model loaded on device: {model.device}")
    print(f"Reference model loaded on device: {ref_model.device}")
    print(f"Initial GPU memory: {get_gpu_memory_usage()}")
    
    # Configure DPO/IPO training
    training_args = DPOConfig(
        # Output & Logging
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        
        # Training parameters
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        
        # DPO/IPO specific parameters
        beta=args.beta,
        loss_type="sigmoid" if args.method == "dpo" else "ipo",
        
        # Batch settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Mixed precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        
        # Memory optimizations
        gradient_checkpointing=args.gradient_checkpointing,
        
        # Dataset configuration
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,  # Keep all columns for DPO
        
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
        
        # Other
        seed=args.seed,
        data_seed=args.seed,
        
        # Hub settings
        push_to_hub=args.push_to_hub,
        hub_model_id=f"smollm-gec-{args.method}" if args.push_to_hub else None,
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Record start time
    start_time = time.time()
    
    # Log memory before training
    print(f"Before training: {get_gpu_memory_usage()}")
    
    # Train the model
    print(f"Starting {args.method.upper()} training...")
    train_result = trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save the model
    print("Saving model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Clean up reference model to free memory
    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    
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
    
    # Save training history
    if hasattr(trainer.state, 'log_history'):
        import json
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print("✅ Training history saved")
    
    # Save results
    config = {
        "method": args.method,
        "sft_model_path": args.sft_model_path,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing
    }
    
    metrics = {
        "bleu_score": bleu_score,
        "train_loss": final_train_loss,
        "eval_loss": eval_loss,
        "training_time": training_time,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "test_samples": len(test_ds)
    }
    
    save_results(args.output_dir, config, metrics)
    
    # Final memory usage
    print(f"After training: {get_gpu_memory_usage()}")
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Method: {args.method.upper()}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print(f"Output Directory: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()