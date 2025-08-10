#!/usr/bin/env python3
"""
Create training_config.json for each experiment and all_experiments_config.json summary
"""

import json
import os
from pathlib import Path
import re

def get_experiment_configs():
    """Define all experiment configurations based on run_experiments.sh"""
    configs = {
        # SFT Padding experiments
        "sft_padding_bs32_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_padding_bs32_lr8e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 8e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_padding_bs64_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "effective_batch_size": 64
        },
        "sft_padding_bs64_lr8e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 8e-5,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "effective_batch_size": 64
        },
        "sft_padding_bs128_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "effective_batch_size": 128
        },
        "sft_padding_bs128_lr8e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 32,
            "learning_rate": 8e-5,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "effective_batch_size": 128
        },
        "sft_padding_bs8_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_padding_bs8_lr8e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 8,
            "learning_rate": 8e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_padding_bs16_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 16,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_padding_bs16_lr8e-05_ep1": {
            "training_type": "sft",
            "method": "padding",
            "batch_size": 16,
            "learning_rate": 8e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        
        # SFT Dataset Packing experiments
        "sft_dataset_packing_bs4_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 4,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_dataset_packing_bs4_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_dataset_packing_bs8_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 8,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_dataset_packing_bs8_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_dataset_packing_bs16_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 16,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_dataset_packing_bs16_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "dataset_packing",
            "batch_size": 16,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        
        # SFT Batch Packing experiments
        "sft_batch_packing_bs4_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 4,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_batch_packing_bs4_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_batch_packing_bs8_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 8,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_batch_packing_bs8_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_batch_packing_bs16_lr3e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 16,
            "learning_rate": 3e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        "sft_batch_packing_bs16_lr5e-05_ep1": {
            "training_type": "sft",
            "method": "batch_packing",
            "batch_size": 16,
            "learning_rate": 5e-5,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1
        },
        
        # DPO experiments
        "dpo_final_model_lr1e-07_ep1": {
            "training_type": "dpo",
            "method": "dpo",
            "batch_size": 16,  # Default from dpo_ipo_train.py
            "learning_rate": 1e-7,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1  # DPO beta parameter
        },
        "dpo_final_model_lr3e-07_ep1": {
            "training_type": "dpo",
            "method": "dpo",
            "batch_size": 16,
            "learning_rate": 3e-7,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1
        },
        "dpo_final_model_lr1e-06_ep1": {
            "training_type": "dpo",
            "method": "dpo",
            "batch_size": 16,
            "learning_rate": 1e-6,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1
        },
        
        # IPO experiments
        "ipo_final_model_lr1e-07_ep1": {
            "training_type": "ipo",
            "method": "ipo",
            "batch_size": 16,
            "learning_rate": 1e-7,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1  # IPO beta parameter
        },
        "ipo_final_model_lr3e-07_ep1": {
            "training_type": "ipo",
            "method": "ipo",
            "batch_size": 16,
            "learning_rate": 3e-7,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1
        },
        "ipo_final_model_lr1e-06_ep1": {
            "training_type": "ipo",
            "method": "ipo",
            "batch_size": 16,
            "learning_rate": 1e-6,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "beta": 0.1
        }
    }
    
    # Add common parameters
    for exp_name, config in configs.items():
        config["model_name"] = "HuggingFaceTB/SmolLM-135M"
        config["optimizer"] = "adamw_torch"
        config["weight_decay"] = 0.01
        config["warmup_ratio"] = 0.1
        config["max_seq_length"] = 512
        config["seed"] = 42
        config["experiment_name"] = exp_name
        
        # Calculate effective batch size if not already set
        if "effective_batch_size" not in config:
            config["effective_batch_size"] = config["batch_size"] * config["gradient_accumulation_steps"]
    
    return configs


def main():
    # Use symlinks instead of hardcoded paths - works on any node
    exp_dir = Path("experiments").resolve()  # Resolves to actual path via symlink
    artifacts_dir = Path("artifacts").resolve()  # Resolves to actual path via symlink
    
    # Get predefined configs
    all_configs = get_experiment_configs()
    
    # Process each experiment
    for exp_name, config in all_configs.items():
        exp_path = exp_dir / exp_name
        
        if not exp_path.exists():
            print(f"Warning: {exp_name} directory not found")
            continue
        
        # Read results.json if available
        results_file = exp_path / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                config["bleu_score"] = results.get("bleu_score")
                config["final_loss"] = results.get("final_loss")
                config["eval_loss"] = results.get("eval_loss")
                config["training_completed"] = True
        else:
            config["training_completed"] = False
        
        # Check if final model exists
        final_model_path = exp_path / "final_model"
        config["has_final_model"] = final_model_path.exists()
        
        # Create individual training_config.json
        config_file = exp_path / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created {config_file}")
    
    # Create all_experiments_config.json
    summary_file = artifacts_dir / "all_experiments_config.json"
    
    # Sort experiments for better readability
    sorted_configs = {}
    # First SFT padding
    for name in sorted([k for k in all_configs.keys() if "sft_padding" in k]):
        sorted_configs[name] = all_configs[name]
    # Then SFT dataset packing
    for name in sorted([k for k in all_configs.keys() if "sft_dataset_packing" in k]):
        sorted_configs[name] = all_configs[name]
    # Then SFT batch packing
    for name in sorted([k for k in all_configs.keys() if "sft_batch_packing" in k]):
        sorted_configs[name] = all_configs[name]
    # Then DPO
    for name in sorted([k for k in all_configs.keys() if "dpo" in k]):
        sorted_configs[name] = all_configs[name]
    # Finally IPO
    for name in sorted([k for k in all_configs.keys() if "ipo" in k]):
        sorted_configs[name] = all_configs[name]
    
    with open(summary_file, "w") as f:
        json.dump(sorted_configs, f, indent=2)
    print(f"\nCreated summary: {summary_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total experiments: {len(all_configs)}")
    print(f"SFT experiments: {sum(1 for k in all_configs if 'sft' in k)}")
    print(f"DPO experiments: {sum(1 for k in all_configs if 'dpo' in k)}")
    print(f"IPO experiments: {sum(1 for k in all_configs if 'ipo' in k)}")
    print(f"Completed experiments: {sum(1 for c in all_configs.values() if c.get('training_completed', False))}")


if __name__ == "__main__":
    main()