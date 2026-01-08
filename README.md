# SmolLM GEC: Style-Preserving Grammar Correction

## Project Overview

This project implements **style-preserving grammatical error correction** using SmolLM-135M with Direct Preference Optimization (DPO). Unlike traditional grammar correction models that rewrite entire sentences, our approach makes **minimal, targeted corrections** - fixing only grammatical errors while preserving your original writing style, vocabulary, and voice.

**Key Achievement**: ~0.50 BLEU score with only 135M parameters, trained on 22+ SFT experiments and 6 DPO/IPO experiments using ~19,000 preference pairs that teach the model to prefer minimal edits over rewrites.

## Model & Checkpoints

| Resource | Link | Description |
|----------|------|-------------|
| **Best Model** | [HuggingFace](https://huggingface.co/DanJZY/SmolLM-135M-GEC-SFT-DPO) | Ready-to-use model (518MB) |
| **Full Checkpoints** | [Google Drive](https://drive.google.com/drive/folders/1fxd_yi4r4crp7UK3tJVKxVPHo9gRcw4z) | All 28 experiments (~68GB) |

### Quick Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("DanJZY/SmolLM-135M-GEC-SFT-DPO")
tokenizer = AutoTokenizer.from_pretrained("DanJZY/SmolLM-135M-GEC-SFT-DPO")

# Example: Fix grammar while preserving style
text = "As the number of people grows, the need of habitable environment is essential."
inputs = tokenizer(f"Fix grammar: {text}", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Motivation

I needed a grammar correction tool that would fix my mistakes without changing how I write. Previous large language models (GPT, Claude, etc.) tend to rewrite sentences entirely, making text sound generic and "AI-generated." This is problematic for professionals who need grammatical accuracy while maintaining their unique voice - lawyers drafting legal documents with precise terminology, academics preserving scholarly tone, or anyone wanting authentic writing that doesn't trigger AI detection. The solution: train a small, efficient model that learns to make minimal corrections through preference learning, understanding something like "using → use" is better than rewriting the entire sentence.

## Example: Style-Preserving vs. Over-Correction

```
Original (with error):
"As the number of people grows, the need of habitable environment is
unquestionably essential."
                                         ↑
                                 grammatical error

✅ Our Model (Style-Preserving):
"As the number of people grows, the need for a habitable environment is
unquestionably essential."
                                         ↑
                            Only fixes "of" → "for"

❌ Typical Model (Over-Correction):
"As population growth continues to grow, the necessity for a habitable
environment becomes undeniably essential."
                            ↑
    Completely rewrites: changes vocabulary, structure, and tone
```

## Real-World Applications

**Legal Industry**: Lawyers drafting contracts, briefs, and legal opinions need precise grammar while maintaining rigorous, formal writing style. Over-correction can alter legal language nuances and change the meaning of carefully crafted clauses.

**ESL/EFL Education**: Language learners need grammar feedback that doesn't rewrite their original ideas. Teachers can differentiate between grammatical errors (which need correction) and stylistic choices (which reflect the student's developing voice).

**Other Applications**: Any domain requiring voice preservation - academic writing, technical documentation, business communications, developer tools, content quality assurance, and professional communications where authenticity matters.

## Technical Highlights

**Custom SmolLM Architecture Implementation**: This project includes a from-scratch implementation of the SmolLM architecture, demonstrating deep understanding of transformer-based language models. The implementation is documented in interactive Jupyter notebooks (see `notebook/` folder) with multiple versions ranging from clean code to comprehensive step-by-step guides. This showcases not just fine-tuning capabilities, but foundational knowledge of model architecture, attention mechanisms, and implementation details.

**Key Technical Contributions**:
- Systematic comparison of efficient training methods for small models (padding, dataset packing, batch packing)
- Novel preference dataset generation for style-preserving grammar correction (~19K pairs using edit distance)
- Empirical evidence that 135M parameter models can achieve competitive GEC performance (BLEU ~0.50)
- Complete hyperparameter search with documented results (22 SFT + 6 DPO/IPO experiments)
- Reproducible training pipeline for resource-constrained environments

**Beyond Simple Fine-Tuning**: This repository demonstrates the complete pipeline from architecture implementation to preference learning optimization, providing both educational value and production-ready training scripts.

---

**📊 Results Included**: This repository contains complete results from 28 experiments (22 SFT + 6 DPO/IPO) achieving BLEU score ~0.50 for grammatical error correction. All configurations, metrics, and analysis are available in the `github_*` directories.

## 🚀 Quick Start

### Explore Results (No Setup Required)
```bash
# View experiment comparison
cat github_artifacts/experiment_results.csv

# Check best model configuration  
cat github_models/best_gec_model/training_config.json

# Browse all experiment hyperparameters
cat github_artifacts/all_experiments_config.json
```

### Prerequisites (For Running Experiments)
- 8 GPUs with 24GB+ VRAM (tested on RTX 3090 and RTX 6000 Ada Generation)
- Conda environment with PyTorch, Transformers, TRL, etc.
- ~200GB free disk space for experiments

### Setup
```bash
# 1. Activate your environment
conda activate SmolLM_gec_project

# 2. Run complete experiment pipeline
./run_experiments.sh

# 3. Or run individual phases
./run_experiments.sh sft        # SFT experiments only
./run_experiments.sh preference # Create preference dataset
./run_experiments.sh dpo        # DPO/IPO experiments
./run_experiments.sh results    # Generate final results

# 4. Prepare results for GitHub (after experiments complete)
./copy_to_github.sh             # Creates github_* directories without model weights
```

#### Optional: FlashAttention 2
If you don't have FlashAttention 2 installed, you can add it (verified on RTX 3090 and RTX 6000 Ada Generation, PyTorch 2.5.1+cu124):

```bash
conda activate SmolLM_gec_project
conda install -n SmolLM_gec_project -y -c nvidia cuda-toolkit=12.4
TORCH_CUDA_ARCH_LIST="8.6" python -m pip install -U flash-attn --no-build-isolation
# Enable in Transformers with: attn_implementation="flash_attention_2"
```

## 📋 Experiment Plan

### Phase 1: SFT Experiments (22 total)
**Padding Method** (10 experiments):
- Batch sizes: 8, 16, 32, 64 (32×2), 128 (32×4) with gradient accumulation
- Learning rates: 5e-5, 8e-5

**Dataset Packing Method** (6 experiments):
- Document-level packing for efficient training
- Batch sizes: 4, 8, 16  
- Learning rates: 3e-5, 5e-5

**Batch Packing Method** (6 experiments):
- Batch-level packing with padding_free or DataCollatorWithFlattening
- Batch sizes: 4, 8, 16
- Learning rates: 3e-5, 5e-5

### Phase 2: Preference Dataset
- Uses best SFT model to generate ~19K preference pairs
- Edit distance annotation for chosen/rejected selection

### Phase 3: DPO/IPO Experiments (6 total)
- Methods: DPO, IPO
- Learning rates: 1e-7, 3e-7, 1e-6
- Starting from best SFT model

## 📊 Expected Results

The pipeline will generate:
- `artifacts/experiment_results.csv` - Complete results table
- `artifacts/summary_statistics.json` - Summary statistics
- `artifacts/*.png` - Comparison plots
- `models/best_gec_model/` - Best performing model

## 🕒 Time Estimates

- **Total Time**: ~2.5-3 hours (with 8 GPU parallelization)
- SFT experiments (22): ~60-75 minutes
- Preference dataset: ~20-30 minutes  
- DPO/IPO experiments (6): ~30-45 minutes
- Results aggregation: ~2-5 minutes

## 📁 Project Structure

```
SmolLM-GEC-SFT-DPO/
├── scripts/                    # Python training scripts
│   ├── utils.py               # Shared utilities
│   ├── sft_train.py           # SFT training
│   ├── create_preference_dataset.py  # Preference data generation
│   ├── dpo_ipo_train.py       # DPO/IPO training
│   └── aggregate_results.py   # Results analysis
├── notebook/                   # Educational Jupyter notebooks
│   ├── README.md              # Detailed guide for all notebooks
│   ├── SmolLM_SFT_DPO_Full.ipynb           # Complete implementation
│   ├── SmolLM_SFT_DPO_Part2.ipynb          # Part 2 only (SFT/DPO)
│   ├── SmolLM_SFT_DPO_Part1_*.ipynb        # Part 1 variants (custom architecture)
│   └── SmolLM_SFT_DPO_Original.ipynb       # Exercise questions only
├── github_experiments/        # All experiment results (GitHub-ready)
├── github_artifacts/          # Analysis results & datasets
├── github_models/             # Best model configuration
├── run_experiments.sh         # Main experiment launcher
├── create_training_configs.py # Generate training configurations
├── copy_to_github.sh          # Prepare GitHub directories
└── SmolLM_SFT_DPO_Implementation_Backup_DataCollatorForCompletionOnlyLM.ipynb  # Backup notebook (fallback)
```

## 📂 Repository Contents

### `github_experiments/` (102MB)
Contains results from all 22 experiments without model weights:
```
github_experiments/
├── sft_padding_bs32_lr5e-05_ep1/    # Example SFT experiment
│   ├── results.json                 # BLEU scores, losses
│   ├── training_config.json         # Complete hyperparameters
│   ├── README.md                    # Experiment description
│   └── final_model/                 # Tokenizer configs only
│       ├── config.json              # Model architecture
│       ├── tokenizer.json           # Tokenization rules
│       └── ...                      # Other config files
├── dpo_final_model_lr3e-07_ep1/     # Example DPO experiment
│   ├── training_history.json        # Detailed training metrics
│   └── ...
└── ... (20 more experiments)
```

### `github_artifacts/` (15MB)
Analysis results and preference dataset:
```
github_artifacts/
├── experiment_results.csv           # Performance comparison table
├── all_experiments_config.json      # Consolidated hyperparameters
├── summary_statistics.json          # Aggregated metrics
├── best_model_info.json            # Best model metadata
├── bleu_by_method.png              # Visualization plots
├── learning_rate_analysis.png
├── sft_bleu_vs_batch_size.png
└── preference_dataset/              # DPO/IPO training data
    ├── preference_dataset.json      # 19K preference pairs
    ├── preference_dataset_human_readable.json
    └── preference_dataset_sample.json
```

### `github_models/` (4.7MB)
Best performing model configuration:
```
github_models/
├── README.md                        # Model documentation
├── best_model_info.json            # Metadata
└── best_gec_model/                 # DPO lr=3e-7, BLEU ~0.50
    ├── training_config.json         # Exact hyperparameters
    ├── config.json                  # Model architecture
    ├── tokenizer.json               # Tokenizer files
    └── ...                          # Other configs
```

### `scripts/`
Core training and evaluation scripts with comprehensive docstrings and argument parsing.

### `notebook/` - Educational Jupyter Notebooks
Interactive notebooks for learning and implementing SFT/DPO training:

**Main Notebooks:**
- `SmolLM_SFT_DPO_Full.ipynb` - Complete implementation with all parts (recommended starting point)
- `SmolLM_SFT_DPO_Part2.ipynb` - SFT/DPO implementation only (skip Part 1)

**Part 1 Variants (Custom SmolLM Architecture):**
- `SmolLM_SFT_DPO_Original.ipynb` - Exercise questions only (no implementation)
- `SmolLM_SFT_DPO_Part1_Reproduce_Raw.ipynb` - Clean implementation without bug comments
- `SmolLM_SFT_DPO_Part1_Debug_Exercise_Solution.ipynb` - With bug-focused comments
- `SmolLM_DPO_SFT_Part1_Debug_Exercise_Solution_Babysit_20250628.ipynb` - Comprehensive step-by-step guide

See `notebook/README.md` for detailed descriptions and learning paths.

**Backup File:**
`SmolLM_SFT_DPO_Implementation_Backup_DataCollatorForCompletionOnlyLM.ipynb` (project root)
- Fallback copy of `SmolLM_SFT_DPO_Full.ipynb`
- Kept at root to avoid breaking anything during reorganization
- **Always try running notebooks in the `notebook/` folder first**
- Only use this backup if something goes wrong with the notebooks folder

### Symlinked Directories (Not in GitHub)
- `experiments/` → `/tmp5/zhuoyuan/smollm_experiments/experiments`
- `artifacts/` → `/tmp5/zhuoyuan/smollm_experiments/artifacts`  
- `models/` → `/tmp5/zhuoyuan/smollm_experiments/models`

These symlinks are used during active experimentation on the compute server.

## 🔧 Manual Usage

### Individual Script Usage

```bash
# SFT Training
python scripts/sft_train.py \
    --method padding \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --base_output_dir experiments

# Create Preference Dataset  
python scripts/create_preference_dataset.py \
    --sft_model_path experiments/sft_padding_bs64_lr5e-5_ep1/final_model \
    --output_dir artifacts/preference_dataset

# DPO Training
python scripts/dpo_ipo_train.py \
    --method dpo \
    --learning_rate 3e-7 \
    --sft_model_path experiments/sft_padding_bs64_lr5e-5_ep1/final_model \
    --preference_dataset_path artifacts/preference_dataset/preference_dataset

# Results Analysis
python scripts/aggregate_results.py \
    --experiments_dir experiments \
    --output_dir artifacts \
    --generate_plots
```

## 🛠️ Utility Scripts

### Generate Training Configurations
Creates comprehensive training configuration documentation:

```bash
# Run after experiments complete
python create_training_configs.py
```

**What it does:**
- Creates `training_config.json` in each experiment folder with:
  - Complete hyperparameters (learning rate, batch size, epochs, etc.)
  - Training method (padding/packing for SFT, DPO/IPO)
  - Model settings (optimizer, warmup, weight decay)
  - Performance metrics (BLEU score, losses)
- Generates `artifacts/all_experiments_config.json` - consolidated table of all 22 experiments
- Essential for reproducibility and understanding the hyperparameter search

**Note**: Already run for this repository - configurations are included in `github_experiments/`.

### Prepare GitHub Directories
Copies experiment results without large model weights:

```bash
# Creates github_* directories from symlinked experiment data
./copy_to_github.sh
```

**What it does:**
- Copies experiment results, configs, and tokenizers to `github_experiments/`
- Copies analysis results and datasets to `github_artifacts/`
- Copies best model configuration to `github_models/`
- Skips model weights (`.safetensors`, `.bin` files) to keep repository size manageable
- Total output: ~122MB (vs ~12GB with weights)

## 🎯 Expected Performance

Based on initial experiments:
- **Baseline (no training)**: BLEU ~0.05
- **SFT**: BLEU ~0.48-0.49
- **DPO/IPO**: BLEU ~0.49-0.50+

## 🐛 Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or enable gradient checkpointing
2. **Missing Dependencies**: Check conda environment activation
3. **No GPUs Detected**: Verify CUDA installation and GPU visibility

### Memory Usage
- RTX 3090 (24GB): Safe batch sizes
  - Padding: up to 128
  - Packing: up to 32
  - DPO/IPO: 4-8 (uses reference model)

## 📈 Monitoring Progress

Watch experiment progress:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check experiment outputs
tail -f experiments/*/logs/train*.log

# View current results
python scripts/aggregate_results.py --experiments_dir experiments --verbose
```

## 🔄 Resuming Experiments

The pipeline is designed to be resumable:
- SFT experiments can be run individually 
- Preference dataset is cached once created
- DPO/IPO experiments can be run independently

## 📝 Citation

If you use this hyperparameter search framework, please cite:

```bibtex
@misc{smollm_gec_hpo,
  title={Hyperparameter Optimization for SmolLM Grammatical Error Correction},
  author={Zhuoyuan Jiang},
  year={2025},
  note={Comprehensive SFT/DPO/IPO hyperparameter search}
}
```

## 🙏 Acknowledgments

Special thanks to **Nima Tajbakhsh** (@ Nvidia) for providing valuable guidance on packing methods and other technical aspects of this project.