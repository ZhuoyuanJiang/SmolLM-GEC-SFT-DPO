# SmolLM GEC Deployment Guide for Lab Server

## 🚀 Quick Deployment Steps

### Step 1: Transfer Code to Server (from your laptop)
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
mkdir -p /tmp5/zhuoyuan/smollm_experiments

# Create symbolic links in project
ln -s /tmp5/zhuoyuan/smollm_experiments experiments
ln -s /tmp5/zhuoyuan/smollm_experiments/models models
ln -s /tmp5/zhuoyuan/smollm_experiments/artifacts artifacts
```

### Step 4: Environment Setup (choose one method)

#### Option A: Create from environment.yml (RECOMMENDED)
```bash
# First, update the environment name in the yml file
sed -i 's/name: sft_dpo_env/name: SmolLM_gec_project/' environment.yml

# Create the environment
conda env create -f environment.yml
conda activate SmolLM_gec_project

# Verify installation
python -c "import torch, transformers, trl; print('Environment ready!')"
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
```

### Step 6: Run Experiments
```bash
# Full pipeline (~2-3 hours)
./run_experiments.sh

# Or individual phases
./run_experiments.sh sft        # Just SFT experiments
./run_experiments.sh preference # Create preference dataset
./run_experiments.sh dpo        # Just DPO/IPO experiments
./run_experiments.sh results    # Generate results table
```

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
- SFT experiments: ~45-60 minutes
- Preference dataset: ~20-30 minutes  
- DPO/IPO experiments: ~30-45 minutes
- Total: ~2-3 hours

## 🎯 Final Output Locations

- Results table: `/tmp5/zhuoyuan/smollm_experiments/artifacts/experiment_results.csv`
- Best model: `/tmp5/zhuoyuan/smollm_experiments/models/best_gec_model/`
- Plots: `/tmp5/zhuoyuan/smollm_experiments/artifacts/*.png`

## 💡 Pro Tips

1. **Run overnight**: Less competition for GPUs
2. **Use screen/tmux**: Experiments continue if SSH disconnects
   ```bash
   screen -S smollm_exp
   ./run_experiments.sh
   # Ctrl+A+D to detach
   # screen -r smollm_exp to reattach
   ```

3. **Test with one experiment first**:
   ```bash
   python scripts/sft_train.py --method padding --batch_size 32 --learning_rate 5e-5
   ```

Good luck with your experiments! 🚀