# SmolLM GEC Hyperparameter Search

Comprehensive hyperparameter optimization for SmolLM-135M on grammatical error correction using SFT, DPO, and IPO methods.

## 🚀 Quick Start

### Prerequisites
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

## 📋 Experiment Plan

### Phase 1: SFT Experiments (12 total)
**Padding Method** (6 experiments):
- Batch sizes: 32, 64 (32×2), 128 (32×4) with gradient accumulation
- Learning rates: 5e-5, 8e-5

**Packing Method** (6 experiments):
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

- **Total Time**: ~2-3 hours (with 8 GPU parallelization)
- SFT experiments: ~45-60 minutes
- Preference dataset: ~20-30 minutes  
- DPO/IPO experiments: ~30-45 minutes
- Results aggregation: ~2-5 minutes

## 📁 Project Structure

```
SmolLM_gec_project/
├── scripts/                    # Python training scripts
│   ├── utils.py               # Shared utilities
│   ├── sft_train.py           # SFT training
│   ├── create_preference_dataset.py  # Preference data generation
│   ├── dpo_ipo_train.py       # DPO/IPO training
│   └── aggregate_results.py   # Results analysis
├── experiments/               # Individual experiment outputs
├── artifacts/                 # Final results and plots
├── models/                    # Best model checkpoints
└── run_experiments.sh         # Main experiment launcher
```

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
  author={Your Name},
  year={2024},
  note={Comprehensive SFT/DPO/IPO hyperparameter search}
}
```