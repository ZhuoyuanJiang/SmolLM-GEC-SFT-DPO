# SmolLM GEC Hyperparameter Search

Comprehensive hyperparameter optimization for SmolLM-135M on grammatical error correction using SFT, DPO, and IPO methods.

**📊 Results Included**: This repository contains complete results from 22 experiments achieving BLEU score ~0.50 for grammatical error correction. All configurations, metrics, and analysis are available in the `github_*` directories.

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
- 8 GPUs with 24GB+ VRAM (RTX 3090 or better)
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
```

#### Optional: FlashAttention 2
If you don't have FlashAttention 2 installed, you can add it (verified on RTX 3090, PyTorch 2.5.1+cu124):

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
├── github_experiments/        # All experiment results (GitHub-ready)
├── github_artifacts/          # Analysis results & datasets
├── github_models/             # Best model configuration
├── run_experiments.sh         # Main experiment launcher
├── create_training_configs.py # Generate training configurations
└── copy_to_github.sh          # Prepare GitHub directories
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