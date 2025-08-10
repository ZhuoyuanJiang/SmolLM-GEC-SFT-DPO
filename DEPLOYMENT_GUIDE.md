# SmolLM GEC Deployment Guide for Lab Server

## 📁 Project Structure (After Deployment)

**Home Directory** (`~/projects/SmolLM-GEC-SFT-DPO/`):
```
SmolLM-GEC-SFT-DPO/
├── scripts/                    # Python training scripts
├── configs/                    # Experiment configurations  
├── experiments/ → /tmp5/zhuoyuan/smollm_experiments/experiments
├── artifacts/   → /tmp5/zhuoyuan/smollm_experiments/artifacts
├── models/      → /tmp5/zhuoyuan/smollm_experiments/models
├── run_experiments.sh          # Main experiment runner
├── requirements.txt
├── environment.yml
└── DEPLOYMENT_GUIDE.md
```

**Local Disk Storage** (`/tmp5/zhuoyuan/smollm_experiments/`):
```
smollm_experiments/
├── experiments/               # Raw experiment outputs (~15GB)
│   ├── sft_padding_bs32_lr5e-5_ep1/
│   ├── dpo_bs16_lr1e-5_ep2/
│   └── ...
├── artifacts/                 # Final results (~1GB)
│   ├── experiment_results.csv
│   ├── training_curves/
│   └── *.png plots
└── models/                    # Model checkpoints (~3-5GB)
    ├── best_gec_model/
    └── intermediate_checkpoints/
```

## ✅ Deployment Status

**✅ COMPLETED STEPS**:
- ✅ Step 1: Code Transfer (via git clone)  
- ✅ Step 2: SSH & Extract (completed)
- ✅ Step 3: Storage Strategy (symlinks working)
- ✅ Step 4: Environment Setup (conda + manual PyTorch)
- ✅ Step 5: HuggingFace Cache (configured)
- ✅ Step 6: Test Setup (all tests passed)

**🚀 READY FOR**: Experiments!

---

## 🚀 Quick Deployment Steps

### Step 1: Transfer Code to Server (from your laptop)

#### Option A: Git Clone (RECOMMENDED)
```bash
# From your server
cd ~/projects
git clone https://github.com/YOUR_USERNAME/SmolLM-GEC-Experiments.git SmolLM-GEC-SFT-DPO
cd SmolLM-GEC-SFT-DPO
```

#### Option B: File Transfer
```bash
# From your laptop (compress for faster transfer)
cd /home/zhuoyuan/CSprojects/Project1_SFT_DPO_OpenSourceDraft_20250803
tar -czf SmolLM_gec_project.tar.gz SmolLM_gec_project/

# Transfer via vllab4 proxy to your projects folder
scp SmolLM_gec_project.tar.gz zhuoyuan@vllab4.ucmerced.edu:~/projects/
```

### Step 2: SSH to Server and Extract
```bash
# SSH through proxy
ssh zhuoyuan@vllab4.ucmerced.edu
ssh vllab6  # or whichever server has GPUs available

# Extract in projects directory
cd ~/projects
tar -xzf SmolLM_gec_project.tar.gz
cd SmolLM_gec_project
```

### Step 3: Setup Storage Strategy
```bash
# Create experiment directory on LOCAL DISK (faster, more space)
# Check available space first
df -h | grep -E "(tmp|ssd|hdd)"

# Based on your df output, /tmp5 has 580GB free (best option)
# Create parallel directory structure on local disk
mkdir -p /tmp5/zhuoyuan/smollm_experiments/experiments
mkdir -p /tmp5/zhuoyuan/smollm_experiments/models  
mkdir -p /tmp5/zhuoyuan/smollm_experiments/artifacts

# Create symbolic links in project (maintains same structure)
ln -s /tmp5/zhuoyuan/smollm_experiments/experiments experiments
ln -s /tmp5/zhuoyuan/smollm_experiments/models models
ln -s /tmp5/zhuoyuan/smollm_experiments/artifacts artifacts

# Verify structure
ls -la | grep -E "(experiments|models|artifacts)"
# Should show: 
# experiments -> /tmp5/zhuoyuan/smollm_experiments/experiments
# models -> /tmp5/zhuoyuan/smollm_experiments/models  
# artifacts -> /tmp5/zhuoyuan/smollm_experiments/artifacts
```

### Step 4: Environment Setup (choose one method)

#### Option A: Create from environment.yml (RECOMMENDED)
```bash
# First, update the environment name in the yml file
sed -i 's/name: sft_dpo_env/name: SmolLM_gec_project/' environment.yml

# Create the environment (may fail on PyTorch CUDA version)
conda env create -f environment.yml
conda activate SmolLM_gec_project

# If PyTorch fails due to CUDA mismatch, install manually:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.53.0 trl==0.19.0 datasets==4.0.0
pip install accelerate evaluate fast-edit-distance pandas matplotlib seaborn tqdm

# Verify installation
python -c "import torch, transformers, trl; print('✅ Environment ready!')"
```

#### Option B: If conda is slow, use pip
```bash
conda create -n SmolLM_gec_project python=3.11 -y
conda activate SmolLM_gec_project
pip install -r requirements.txt
```

#### Option C: Manual installation (if above fail)
```bash
conda create -n SmolLM_gec_project python=3.11 -y
conda activate SmolLM_gec_project

# Install PyTorch for CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers==4.53.0 trl==0.19.0 datasets==4.0.0
pip install accelerate evaluate fast-edit-distance
pip install pandas matplotlib seaborn tqdm
```

### Step 5: Configure HuggingFace Cache
```bash
# IMPORTANT: Prevent HF from filling your 100GB home quota
export HF_HOME=/tmp5/zhuoyuan/hf_cache
export TRANSFORMERS_CACHE=/tmp5/zhuoyuan/hf_cache
export HF_DATASETS_CACHE=/tmp5/zhuoyuan/hf_cache

# Add to your .bashrc to make permanent
echo 'export HF_HOME=/tmp5/zhuoyuan/hf_cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/tmp5/zhuoyuan/hf_cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/tmp5/zhuoyuan/hf_cache' >> ~/.bashrc

# Create cache directory
mkdir -p /tmp5/zhuoyuan/hf_cache
```

### Step 6: Test Setup
```bash
# Test 1: Library imports
python -c "import torch, transformers, trl, datasets, accelerate; print('✅ All libraries imported successfully')"

# Test 2: CUDA and GPU detection
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test 3: Script accessibility
python scripts/sft_train.py --help

# Test 4: HuggingFace cache configuration
python -c "
import os
print('HuggingFace cache locations:')
print(f'HF_HOME: {os.getenv(\"HF_HOME\", \"Not set\")}')
print(f'TRANSFORMERS_CACHE: {os.getenv(\"TRANSFORMERS_CACHE\", \"Not set\")}')
print(f'HF_DATASETS_CACHE: {os.getenv(\"HF_DATASETS_CACHE\", \"Not set\")}')
"

# Test 5: Storage structure
ls -la | grep -E "(experiments|models|artifacts)"
echo "Storage locations:"
echo "Experiments: $(readlink experiments)"
echo "Models: $(readlink models)" 
echo "Artifacts: $(readlink artifacts)"

# Test 6: Model loading to cache (optional)
python -c "
from transformers import AutoTokenizer
print('Testing model download to cache...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
print('✅ Model loaded successfully!')
print('Cache should be in /tmp5/zhuoyuan/hf_cache/')
"
```

**Expected Results:**
- ✅ All libraries imported successfully
- PyTorch 2.5.1+cu124, CUDA 12.4 available, 8 RTX 3090 GPUs detected
- SFT script shows help menu with training parameters
- All cache paths point to `/tmp5/zhuoyuan/hf_cache`
- Symbolic links point to `/tmp5/zhuoyuan/smollm_experiments/...`
- Small model downloads successfully to cache

### Step 7: Run Experiments

#### Option 1: Quick Test (RECOMMENDED FIRST)
```bash
# Test standard padding first (5-10 minutes)
CUDA_VISIBLE_DEVICES=0 python scripts/sft_train.py \
    --method padding --batch_size 16 --learning_rate 5e-5 \
    --num_epochs 1

# Test batch_packing support (optional, 5-10 minutes)
CUDA_VISIBLE_DEVICES=1 python scripts/sft_train.py \
    --method batch_packing --batch_size 8 --learning_rate 5e-5 \
    --num_epochs 1
```
**Expected Results:**
- Model downloads to `/tmp5/zhuoyuan/hf_cache/` 
- Training progress bars and loss decreasing
- Final models saved to:
  - `experiments/sft_padding_bs16_lr5e-5_ep1/` (padding test)
  - `experiments/sft_batch_packing_bs8_lr5e-5_ep1/` (batch_packing test)
- Each test should complete in 5-10 minutes

#### Option 2: Full Pipeline (All-in-One)
```bash
# Complete hyperparameter search (~2-3 hours)
screen -S smollm_exp
./run_experiments.sh
# Press Ctrl+A+D to detach, screen -r smollm_exp to reattach
```
**Expected Results:**
- 22 SFT experiments (16 base + 6 batch_packing) → Best model selected
- Preference dataset generation (~19K pairs)
- 6 DPO/IPO experiments → Final model ranking
- Results table in `artifacts/experiment_results.csv`
- Best model in `models/best_gec_model/`

#### Option 3: Staged Approach (RECOMMENDED)
```bash
# Phase 1: SFT experiments (~1.25 hours)
./run_experiments.sh sft
# Check: ls experiments/ should show 22 SFT model directories

# Phase 2: Preference dataset (~30 minutes)  
./run_experiments.sh preference
# Check: ls experiments/ should show preference_dataset.json

# Phase 3: DPO/IPO experiments (~1 hour)
./run_experiments.sh dpo
# Check: ls experiments/ should show 6 DPO/IPO model directories

# Phase 4: Results analysis (~5 minutes)
./run_experiments.sh results
# Check: artifacts/experiment_results.csv should exist
```
**Expected Results by Phase:**
- **Phase 1**: 22 experiment directories, best SFT model identified
- **Phase 2**: ~19K preference pairs, dataset validation metrics
- **Phase 3**: 6 additional experiment directories, final model comparison
- **Phase 4**: Complete results CSV, performance plots, best model selection

## ⚠️ Important Storage Notes

1. **Home Directory Limit**: Only 100GB total
   - Code only (~30MB) ✓
   - NO models, checkpoints, or datasets here

2. **Local Disk Usage**:
   - `/tmp5`: 580GB free (recommended)
   - `/tmp4`: 185GB free (backup option)
   - Avoid disks >90% full

3. **Storage Estimates**:
   - Each SFT model: ~500MB
   - Preference dataset: ~2GB
   - Total experiments: ~15-20GB

## 🔧 Path Adjustments Needed

The code uses relative paths, so minimal adjustments needed:

1. **In run_experiments.sh** (already handled with symlinks):
   ```bash
   EXPERIMENTS_DIR="experiments"  # → /tmp5/zhuoyuan/smollm_experiments
   ARTIFACTS_DIR="artifacts"      # → /tmp5/zhuoyuan/smollm_experiments/artifacts
   MODELS_DIR="models"            # → /tmp5/zhuoyuan/smollm_experiments/models
   ```

2. **No Python code changes needed** - all paths are relative!

## 📊 Monitoring Resources

```bash
# Watch GPU usage (in another terminal)
watch -n 1 nvidia-smi

# Monitor disk usage
df -h /tmp5

# Check your home quota
du -sh ~/
```

## 🚨 Troubleshooting

### Issue: "No space left on device"
```bash
# Check where space is being used
du -sh ~/.cache/huggingface  # This is the usual culprit
# Clear if needed
rm -rf ~/.cache/huggingface/*
```

### Issue: CUDA version mismatch
```bash
# Your server has CUDA 12.4, code expects 12.1
# PyTorch should auto-select compatible version
# If issues, reinstall PyTorch:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Issue: Someone needs your GPUs (PhD student priority)
```bash
# Save progress and gracefully exit
# Results are saved after each experiment
# You can resume with specific phases later
```

## 📈 Expected Timeline

With 8 GPUs available:
- SFT experiments (22): ~60-75 minutes
- Preference dataset: ~20-30 minutes  
- DPO/IPO experiments: ~30-45 minutes
- Total: ~2.5-3 hours

## 🎯 Final Output Locations

- Results table: `/tmp5/zhuoyuan/smollm_experiments/artifacts/experiment_results.csv`
- Best model: `/tmp5/zhuoyuan/smollm_experiments/models/best_gec_model/`
- Plots: `/tmp5/zhuoyuan/smollm_experiments/artifacts/*.png`

## 💡 Pro Tips

1. **Run overnight**: Less competition for GPUs
2. **Use screen/tmux**: Experiments continue if SSH disconnects
   ```bash
   screen -S smollm_exp
   conda activate SmolLM_gec_project 
   ./run_experiments.sh
   # Ctrl+A+D to detach
   # screen -r smollm_exp to reattach
   ```

3. **Test with one experiment first**:
   ```bash
   python scripts/sft_train.py --method padding --batch_size 32 --learning_rate 5e-5
   ```

## 🧹 Cleanup (When You're Done)

**Complete cleanup** (removes all experiment data):
```bash
# Remove all experiment data from local disk (15-20GB)
rm -rf /tmp5/zhuoyuan/smollm_experiments/

# Remove HuggingFace cache (5-10GB)  
rm -rf /tmp5/zhuoyuan/hf_cache/

# Remove conda environment (Optional)
conda env remove -n SmolLM_gec_project

# Remove project code (optional)
rm -rf ~/projects/SmolLM-GEC-SFT-DPO/
```

**Partial cleanup** (keep results, remove large files):
```bash
# Keep results CSV and plots, remove model checkpoints
rm -rf /tmp5/zhuoyuan/smollm_experiments/models/
rm -rf /tmp5/zhuoyuan/smollm_experiments/experiments/*/checkpoints/

# Clear HuggingFace cache
rm -rf /tmp5/zhuoyuan/hf_cache/
```

**Check what you're using**:
```bash
du -sh /tmp5/zhuoyuan/  # Total space used
ls -la ~/projects/SmolLM-GEC-SFT-DPO/  # Verify symlink structure
```

Good luck with your experiments! 🚀