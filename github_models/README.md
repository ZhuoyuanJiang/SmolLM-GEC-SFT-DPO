# Best GEC Model

## Overview

This folder contains documentation(including the configuration and tokenizer files) about the trained models. The actual model weights (`.safetensors` files) are stored on the remote server and not included in this repository due to their size (~500MB each).

## 📊 Best Model Performance

**Model**: `dpo_final_model_lr3e-07_ep1`
- **Training Method**: DPO (Direct Preference Optimization)
- **Base Model**: SmolLM-135M (fine-tuned with SFT first)
- **BLEU Score**: ~0.49-0.50
- **Learning Rate**: 3e-7
- **Batch Size**: 16
- **Beta**: 0.1

See `best_gec_model/training_config.json` for complete hyperparameters.

## 📁 Directory Contents

### `best_gec_model/`
Contains all configuration files needed to initialize the model:
- `config.json` - Model architecture configuration
- `tokenizer_config.json` - Tokenizer settings
- `tokenizer.json` - Tokenization rules
- `vocab.json` - Vocabulary (49,152 tokens)
- `merges.txt` - BPE merge rules
- `special_tokens_map.json` - Special tokens mapping
- `generation_config.json` - Generation settings
- `added_tokens.json` - Additional tokens
- `training_config.json` - Training hyperparameters used

### `best_model_info.json`
Metadata about which experiment produced the best model.

## ⚠️ Model Weights Not Included

The actual model weights (`model.safetensors`, ~538MB) are not included in this repository due to size constraints. 

### Accessing Full Model Weights

#### Option 1: Contact Repository Owner
For access to the trained model weights, please contact the repository owner through GitHub issues or email.

#### Option 2: HuggingFace Hub (To Be Updated)
The model weights may be uploaded to HuggingFace Hub in the future. Check back for updates or watch this repository for announcements.

## 🔧 Usage

### Loading Configuration and Tokenizer (from this repo)
```python
from transformers import AutoTokenizer, AutoConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("github_models/best_gec_model/")

# Load configuration
config = AutoConfig.from_pretrained("github_models/best_gec_model/")

# View training hyperparameters
import json
with open("github_models/best_gec_model/training_config.json") as f:
    training_params = json.load(f)
print(training_params)
```

### Initializing Model from Config (requires training)
```python
from transformers import AutoModelForCausalLM

# Initialize model with random weights
model = AutoModelForCausalLM.from_config(config)
# Note: This model has random weights and needs to be trained!
```

### Inference Example (with trained weights)
```python
def correct_text(text, model, tokenizer):
    prompt = f"Correct the following text: {text}\nCorrected: "
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Training Pipeline

This model was produced through a three-stage training pipeline:

1. **SFT (Supervised Fine-Tuning)**: 16 experiments with different hyperparameters
2. **Preference Dataset Generation**: Created ~19K preference pairs using the best SFT model
3. **DPO/IPO Training**: 6 experiments, with DPO lr=3e-7 performing best

## 📊 Comparison with Other Models

| Model | BLEU Score | Method |
|-------|------------|--------|
| Baseline (no training) | 0.05 | - |
| Best SFT | 0.49 | Padding, bs=16, lr=5e-5 |
| **Best DPO (this model)** | **0.49-0.50** | **DPO, lr=3e-7** |
| Best IPO | 0.48 | IPO, lr=3e-7 |

For complete results, see `../github_artifacts/experiment_results.csv`

## 🔄 Reproducing Results

To reproduce this model:

1. Run SFT training:
```bash
python scripts/sft_train.py --method padding --batch_size 16 --learning_rate 5e-5
```

2. Generate preference dataset:
```bash
python scripts/create_preference_dataset.py --sft_model_path [best_sft_path]
```

3. Run DPO training:
```bash
python scripts/dpo_ipo_train.py --method dpo --learning_rate 3e-7 --sft_model_path [best_sft_path]
```

Or use the complete pipeline:
```bash
./run_experiments.sh
```

## 📝 Citation

If you use this model, please cite:
```bibtex
@misc{smollm_gec_best,
  title={SmolLM-135M Fine-tuned for Grammatical Error Correction},
  author={Your Name},
  year={2024},
  note={Best model from comprehensive SFT/DPO/IPO hyperparameter search}
}
```