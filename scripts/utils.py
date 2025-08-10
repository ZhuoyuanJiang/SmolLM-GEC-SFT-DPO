"""
Shared utility functions for SmolLM GEC experiments
"""

import torch
import random
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoTokenizer
import evaluate
from tqdm import tqdm
from typing import Dict, Any, List, Tuple


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seeds set to {seed} for reproducibility")


def setup_tokenizer_once(model, tokenizer):
    """
    One-time tokenizer setup for all SFT/DPO/IPO experiments.
    This ensures pad_token != eos_token to fix repetitive generation.
    Also adds [SEP] as a special token for reliable response separation.
    Maintains dtype consistency after adding tokens.
    """
    print("=== Setting up tokenizer ===")

    # Store the model's dtype for consistency
    model_dtype = model.dtype

    # Check if proper pad token already exists
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        print(f"✓ Pad token already properly configured: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    else:
        # Add [PAD] token - this is the safest approach
        print("Adding [PAD] token to vocabulary...")
        num_added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Added {num_added} new tokens")

        # Resize model embeddings to match tokenizer
        print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

        # Ensure embeddings stay in correct dtype
        model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.to(model_dtype)
        if hasattr(model, 'lm_head'):
            model.lm_head.weight.data = model.lm_head.weight.data.to(model_dtype)

        # Initialize the new PAD embedding as zeros
        with torch.no_grad():
            model.get_input_embeddings().weight[-1].zero_()

    # Check if response separator token [SEP] already exists
    if '[SEP]' not in tokenizer.get_vocab():
        print("\nAdding [SEP] separator token...")
        old_vocab_size = len(tokenizer)

        # Add the special token
        if 'additional_special_tokens' in tokenizer.special_tokens_map:
            existing = tokenizer.special_tokens_map['additional_special_tokens']
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing + ['[SEP]']
            })
        else:
            tokenizer.add_special_tokens({
                'additional_special_tokens': ['[SEP]']
            })

        # Resize if needed
        if len(tokenizer) > old_vocab_size:
            model.resize_token_embeddings(len(tokenizer))

            # Ensure embeddings stay in correct dtype
            model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.to(model_dtype)
            if hasattr(model, 'lm_head'):
                model.lm_head.weight.data = model.lm_head.weight.data.to(model_dtype)

            print(f"Added [SEP] token, new vocab size: {len(tokenizer)}")

            # Initialize [SEP] embedding as mean of existing embeddings
            with torch.no_grad():
                mean_embedding = model.get_input_embeddings().weight[:-1].mean(dim=0)
                model.get_input_embeddings().weight[-1] = mean_embedding

    # Set padding side for causal LM
    tokenizer.padding_side = "left"

    # Update model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # Verification
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
    print(f"\n✓ Setup complete:")
    print(f"  - PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  - SEP token: '[SEP]' (ID: {sep_token_id})")
    print(f"  - EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  - PAD != EOS: {tokenizer.pad_token_id != tokenizer.eos_token_id} ✓")
    print(f"  - Model dtype: {model.dtype}")  # Added dtype check

    return model, tokenizer


def format_text(example: Dict[str, str], tokenizer) -> str:
    """
    Format dataset example for training.
    Since the instruction is already in src, we just need clear separation
    between the full instruction+incorrect_text and the corrected_text
    """
    return f"{example['src']}[SEP]{example['tgt']}{tokenizer.eos_token}"


def format_dataset(dataset, tokenizer):
    """Apply formatting to dataset"""
    return dataset.map(lambda x: {"text": format_text(x, tokenizer)})


def format_inference_text(text: str) -> str:
    """Format input for inference to match training format"""
    return f"{text}[SEP]"


def extract_correction_from_output(full_output: str, tokenizer) -> str:
    """Extract the correction part from model output"""
    # Remove EOS and PAD tokens manually
    cleaned_output = full_output.replace(tokenizer.eos_token, "").strip()
    if tokenizer.pad_token:
        cleaned_output = cleaned_output.replace(tokenizer.pad_token, "").strip()

    # Now extract correction after [SEP]
    if "[SEP]" in cleaned_output:
        parts = cleaned_output.split("[SEP]", 1)
        if len(parts) > 1:
            return parts[1].strip()

    # Fallback: return cleaned output
    return cleaned_output


def evaluate_model(model, tokenizer, test_dataset, batch_size: int = 8, show_progress: bool = True) -> float:
    """
    Evaluate model on the test dataset and return BLEU score.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        test_dataset: The UNFORMATTED test dataset (with 'src' and 'tgt' fields)
        batch_size: Batch size for evaluation (default 8)
        show_progress: Whether to show progress bar
    
    Returns:
        BLEU score (float)
    """
    preds = []
    targets = []

    # Put model in eval mode
    model.eval()
    
    # Create progress bar if needed
    iterator = tqdm(test_dataset, desc="Evaluating") if show_progress else test_dataset

    for i, example in enumerate(iterator):
        # Get input from 'src' field
        input_text = example['src']

        # Format exactly as in training
        formatted_input = f"{input_text}[SEP]"

        # Tokenize
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(model.device)

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract prediction
        prediction = extract_correction_from_output(full_output, tokenizer)

        # Get ground truth from 'tgt' field
        target = example['tgt'].strip()

        # Add to lists
        preds.append(prediction)
        targets.append([target])  # BLEU expects list of lists

        # Print progress with intermediate BLEU every 100 samples
        if show_progress and (i + 1) % 100 == 0:
            # Calculate intermediate BLEU score
            bleu = evaluate.load("bleu")
            intermediate_score = bleu.compute(
                predictions=preds,
                references=targets
            )["bleu"]
            print(f"\nProcessed {i + 1}/{len(test_dataset)} samples... BLEU: {intermediate_score:.4f}")

    # Put model back in train mode
    model.train()

    # Calculate final BLEU score
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=preds, references=targets)
    return results["bleu"]


def save_results(
    output_dir: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    system_info: Dict[str, str] = None
) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        output_dir: Directory to save results
        config: Experiment configuration
        metrics: Experiment metrics (BLEU score, losses, etc.)
        system_info: System information (GPU, CUDA version, etc.)
    """
    if system_info is None:
        system_info = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "timestamp": datetime.now().isoformat()
        }
    
    results = {
        "config": config,
        "metrics": metrics,
        "system": system_info
    }
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(results_path, "r") as f:
        return json.load(f)


def get_gpu_memory_usage() -> str:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "No GPU available"


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """Create output directory for experiment"""
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir