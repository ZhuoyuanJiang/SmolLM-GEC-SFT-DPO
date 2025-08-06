#!/usr/bin/env python3
"""
Create Preference Dataset for DPO/IPO Training
Generates chosen/rejected pairs using edit distance annotation
"""

import argparse
import os
import json
import gc
from datetime import datetime
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from fast_edit_distance import edit_distance
from tqdm.auto import tqdm

# Import our utility functions
from utils import set_seed, setup_tokenizer_once, save_results, get_gpu_memory_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Create preference dataset for DPO/IPO training")
    
    # Model arguments
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to the fine-tuned SFT model")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="grammarly/coedit",
                        help="Dataset to use for preference generation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preference dataset")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    
    # Generation configurations
    parser.add_argument("--temp1", type=float, default=0.3,
                        help="Temperature for first generation config (conservative)")
    parser.add_argument("--temp2", type=float, default=0.7,
                        help="Temperature for second generation config (diverse)")
    parser.add_argument("--top_p1", type=float, default=0.7,
                        help="Top-p for first generation config")
    parser.add_argument("--top_p2", type=float, default=0.9,
                        help="Top-p for second generation config")
    parser.add_argument("--rep_penalty1", type=float, default=1.05,
                        help="Repetition penalty for first config")
    parser.add_argument("--rep_penalty2", type=float, default=1.15,
                        help="Repetition penalty for second config")
    
    # Filtering parameters
    parser.add_argument("--min_edit_distance_diff", type=int, default=2,
                        help="Minimum edit distance difference between chosen/rejected")
    parser.add_argument("--max_length_ratio", type=float, default=3.0,
                        help="Maximum length ratio between chosen/rejected")
    parser.add_argument("--min_length_ratio", type=float, default=0.33,
                        help="Minimum length ratio between chosen/rejected")
    
    # Progress saving
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Save intermediate results every N examples")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for generation")
    
    return parser.parse_args()


def load_gec_dataset(dataset_name: str):
    """Load and filter GEC dataset"""
    print(f"Loading dataset: {dataset_name}")
    
    # Load training split for preference dataset generation
    full_train_ds = load_dataset(dataset_name, split="train")
    
    # Filter for GEC task only
    train_ds = full_train_ds.filter(lambda example: example['task'] == 'gec')
    
    print(f"GEC training dataset size: {len(train_ds)}")
    return train_ds


def format_input_for_generation(text: str) -> str:
    """Format input text to match training format"""
    return f"{text}[SEP]"


def extract_correction_from_output(full_output: str, tokenizer, formatted_input: str) -> str:
    """Extract the correction part from model output"""
    # Remove EOS and PAD tokens manually
    cleaned_output = full_output.replace(tokenizer.eos_token, "").strip()
    if tokenizer.pad_token:
        cleaned_output = cleaned_output.replace(tokenizer.pad_token, "").strip()

    # Extract correction after [SEP]
    if "[SEP]" in cleaned_output:
        parts = cleaned_output.split("[SEP]", 1)
        if len(parts) > 1:
            return parts[1].strip()

    # Fallback: remove the input part
    return cleaned_output.replace(formatted_input, "").strip()


def generate_variant_batch(model, tokenizer, prompts: List[str], generation_config: dict) -> List[str]:
    """Generate variants for a batch of prompts with given configuration"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode all outputs
    batch_results = []
    for i, output in enumerate(outputs):
        full_output = tokenizer.decode(output, skip_special_tokens=False)
        correction = extract_correction_from_output(full_output, tokenizer, prompts[i])
        batch_results.append(correction)

    return batch_results


def calculate_preference(pred1: str, pred2: str, ground_truth: str) -> Tuple[str, str, int, int]:
    """Calculate which prediction is better based on edit distance"""
    dist1 = edit_distance(pred1, ground_truth)
    dist2 = edit_distance(pred2, ground_truth)

    if dist1 < dist2:
        return pred1, pred2, dist1, dist2
    else:
        return pred2, pred1, dist2, dist1


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset
    train_ds = load_gec_dataset(args.dataset_name)
    
    # Load SFT model and tokenizer
    print(f"Loading SFT model from: {args.sft_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    
    # Load tokenizer (should be in the same directory as the model)
    tokenizer_path = args.sft_model_path
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        # Fallback to base model tokenizer
        print("Warning: Using base model tokenizer - ensure it matches training setup")
        tokenizer_path = "HuggingFaceTB/SmolLM-135M"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Setup tokenizer (should already be configured, but ensure consistency)
    model, tokenizer = setup_tokenizer_once(model, tokenizer)
    
    # Ensure model is in eval mode
    model.eval()
    print(f"Model loaded on device: {model.device}")
    print(f"Initial GPU memory: {get_gpu_memory_usage()}")
    
    # Generation configurations
    generation_config_1 = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temp1,
        "do_sample": True,
        "top_p": args.top_p1,
        "repetition_penalty": args.rep_penalty1,
    }
    
    generation_config_2 = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temp2,
        "do_sample": True,
        "top_p": args.top_p2,
        "repetition_penalty": args.rep_penalty2,
    }
    
    # Initialize data structures
    preference_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    stats = {
        "total_processed": 0,
        "successful": 0,
        "skipped_identical": 0,
        "skipped_empty": 0,
        "skipped_small_diff": 0,
        "skipped_length_ratio": 0,
        "total_chosen_dist": 0,
        "total_rejected_dist": 0
    }
    
    print(f"\nCreating preference dataset from {len(train_ds)} training examples...")
    print(f"Batch size: {args.batch_size}, Min edit distance diff: {args.min_edit_distance_diff}")
    start_time = datetime.now()
    
    # Process dataset in batches
    for batch_start in tqdm(range(0, len(train_ds), args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(train_ds))
        current_batch_size = batch_end - batch_start
        
        # Prepare batch data
        batch_prompts = []
        batch_examples = []
        
        for idx in range(batch_start, batch_end):
            example = train_ds[idx]
            prompt = format_input_for_generation(example['src'])
            batch_prompts.append(prompt)
            batch_examples.append(example)
        
        try:
            # Generate both configs for the entire batch
            batch_pred1 = generate_variant_batch(model, tokenizer, batch_prompts, generation_config_1)
            batch_pred2 = generate_variant_batch(model, tokenizer, batch_prompts, generation_config_2)
            
            # Process each example in the batch
            for i in range(current_batch_size):
                stats["total_processed"] += 1
                
                pred1 = batch_pred1[i]
                pred2 = batch_pred2[i]
                ground_truth = batch_examples[i]['tgt']
                
                # Apply filters
                if not pred1 or not pred2:
                    stats["skipped_empty"] += 1
                    continue
                
                if pred1 == pred2:
                    stats["skipped_identical"] += 1
                    continue
                
                # Calculate preferences
                chosen, rejected, chosen_dist, rejected_dist = calculate_preference(
                    pred1, pred2, ground_truth
                )
                
                # Skip if edit distance difference is too small
                if abs(chosen_dist - rejected_dist) < args.min_edit_distance_diff:
                    stats["skipped_small_diff"] += 1
                    continue
                
                # Skip if length ratio is extreme
                if len(rejected) > 0:
                    length_ratio = len(chosen) / len(rejected)
                    if length_ratio > args.max_length_ratio or length_ratio < args.min_length_ratio:
                        stats["skipped_length_ratio"] += 1
                        continue
                else:
                    stats["skipped_empty"] += 1
                    continue
                
                # Add to dataset
                preference_data["prompt"].append(batch_examples[i]['src'])
                preference_data["chosen"].append(chosen)
                preference_data["rejected"].append(rejected)
                
                stats["successful"] += 1
                stats["total_chosen_dist"] += chosen_dist
                stats["total_rejected_dist"] += rejected_dist
                
                # Save intermediate results
                if stats['total_processed'] % args.save_interval == 0 and stats['total_processed'] > 0:
                    print(f"Saving intermediate results at {stats['total_processed']} examples...")
                    temp_dataset = Dataset.from_dict(preference_data)
                    temp_path = os.path.join(args.output_dir, f"temp_{stats['total_processed']}")
                    temp_dataset.save_to_disk(temp_path)
        
        except Exception as e:
            print(f"\nError processing batch {batch_start}-{batch_end}: {str(e)}")
            continue
        
        # Periodic maintenance
        if (batch_start // args.batch_size) % 2 == 0 and batch_start > 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print progress update every 20 batches
        if (batch_start // args.batch_size) % 20 == 0 and batch_start > 0:
            elapsed = datetime.now() - start_time
            success_rate = stats['successful'] / stats['total_processed'] * 100 if stats['total_processed'] > 0 else 0
            print(f"\nProgress: {stats['total_processed']}/{len(train_ds)} - "
                  f"Elapsed: {elapsed.total_seconds()//60:.0f}m {elapsed.total_seconds()%60:.0f}s - "
                  f"Success rate: {success_rate:.1f}%")
    
    # Final statistics
    elapsed = datetime.now() - start_time
    print(f"\n✅ Dataset creation completed in {elapsed.total_seconds()//60:.0f}m {elapsed.total_seconds()%60:.0f}s!")
    
    print("\n" + "="*60)
    print("PREFERENCE DATASET STATISTICS:")
    print("="*60)
    print(f"Total examples processed: {stats['total_processed']}")
    print(f"Successful examples: {stats['successful']}")
    print(f"Skipped (identical outputs): {stats['skipped_identical']}")
    print(f"Skipped (empty predictions): {stats['skipped_empty']}")
    print(f"Skipped (small edit distance diff): {stats['skipped_small_diff']}")
    print(f"Skipped (extreme length ratio): {stats['skipped_length_ratio']}")
    print("-"*60)
    if stats['successful'] > 0:
        print(f"Average chosen edit distance: {stats['total_chosen_dist']/stats['successful']:.2f}")
        print(f"Average rejected edit distance: {stats['total_rejected_dist']/stats['successful']:.2f}")
        print(f"Success rate: {stats['successful']/stats['total_processed']*100:.1f}%")
    print("="*60)
    
    # Save final dataset
    preference_dataset = Dataset.from_dict(preference_data)
    print(f"\nFinal dataset size: {len(preference_dataset)}")
    
    preference_dataset.save_to_disk(os.path.join(args.output_dir, "preference_dataset"))
    print(f"✅ Dataset saved to {os.path.join(args.output_dir, 'preference_dataset')}")
    
    # Save JSON version
    with open(os.path.join(args.output_dir, "preference_dataset.json"), "w") as f:
        json.dump(preference_data, f, indent=2)
    print(f"✅ JSON saved to {os.path.join(args.output_dir, 'preference_dataset.json')}")
    
    # Always save human-readable version for inspection and debugging
    human_readable = []
    for i in range(len(preference_data['prompt'])):
        human_readable.append({
            "entry_number": i,
            "prompt": preference_data['prompt'][i],
            "chosen": preference_data['chosen'][i],
            "rejected": preference_data['rejected'][i]
        })
    
    # Save full human-readable version
    with open(os.path.join(args.output_dir, "preference_dataset_human_readable.json"), "w") as f:
        json.dump(human_readable, f, indent=2)
    print(f"✅ Human-readable version saved to {os.path.join(args.output_dir, 'preference_dataset_human_readable.json')}")
    
    # Save a sample for quick quality check (first 100 examples)
    sample_size = min(100, len(human_readable))
    sample_data = human_readable[:sample_size]
    with open(os.path.join(args.output_dir, "preference_dataset_sample.json"), "w") as f:
        json.dump(sample_data, f, indent=2)
    print(f"✅ Sample ({sample_size} examples) saved to {os.path.join(args.output_dir, 'preference_dataset_sample.json')}")
    
    # Save configuration and stats
    config = {
        "sft_model_path": args.sft_model_path,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "generation_configs": {
            "config_1": generation_config_1,
            "config_2": generation_config_2
        },
        "filtering": {
            "min_edit_distance_diff": args.min_edit_distance_diff,
            "max_length_ratio": args.max_length_ratio,
            "min_length_ratio": args.min_length_ratio
        },
        "seed": args.seed
    }
    
    metrics = {
        "final_dataset_size": len(preference_dataset),
        "processing_time": elapsed.total_seconds(),
        **stats
    }
    
    save_results(args.output_dir, config, metrics)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n✅ Memory cleaned. Final GPU: {get_gpu_memory_usage()}")


if __name__ == "__main__":
    main()