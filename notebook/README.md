# SmolLM SFT/DPO Training Notebooks

This directory contains Jupyter notebooks for learning and implementing Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) for grammatical error correction using SmolLM-135M.

## Quick Start

**Just want to see the complete project?**
→ Start with **`SmolLM_SFT_DPO_Full.ipynb`** - This is the ultimate notebook with full implementation of all parts.

**Want to learn step-by-step?**
→ See the [Learning Paths](#learning-paths) section below.

## Exercise Structure

This exercise consists of 2 parts:

- **Part 1**: Implementing custom SmolLM architecture (optional, isolated from Part 2)
  - The debugging exercise was motivated by implementing this architecture
  - I debugged extensively during implementation and documented the process as an educational exercise
- **Part 2**: Complete SFT/DPO implementation
  - Combines the original DPO implementation (Part 2) and DPO variants like IPO (Part 3)

**Important**: Part 1 is completely independent from Part 2. You can skip Part 1 entirely if you're only interested in the SFT/DPO implementation.

## Notebook Descriptions

### 🎯 Main Notebook (Start Here for Full Project)

#### `SmolLM_SFT_DPO_Full.ipynb`
**THE ULTIMATE NOTEBOOK** - Contains the complete implementation of all parts.

- ✅ Full implementation of Part 1 (bug fixes) + Part 2 (SFT/DPO)
- ✅ All results and visualizations
- ✅ **This is the notebook you should run to get all the results**
- ✅ Best for understanding the complete project without diving into specific details

**Use this if**: You want to see the full project, run experiments, and get comprehensive results.

---

### 📚 Part 1 Notebooks (Custom SmolLM Architecture Implementation - Optional)

Part 1 focuses on implementing custom SmolLM architecture with debugging exercises. **These are completely optional** - Part 2 works independently.

#### `SmolLM_SFT_DPO_Original.ipynb`
**Original exercise questions only** - No implementation.

- ❌ No implementation code
- ✅ Only contains the problem statements and questions
- ✅ Useful for understanding the scope before diving into code
- ✅ **Best for implementing your own solution from scratch**

**Use this if**: You want to understand what the exercise asks for without being influenced by the implementation, or you want to implement it yourself.

---

#### `SmolLM_SFT_DPO_Part1_Reproduce_Raw.ipynb`
**Part 1 implemented** - No bug comments, just correct code.

- ✅ Complete Part 1 implementation
- ⚠️ No comments around bugs - just the correct code
- ✅ Minimal bug fix without explanation

**Use this if**: You want to see the clean, corrected implementation without debugging explanations.

---

#### `SmolLM_SFT_DPO_Part1_Debug_Exercise_Solution.ipynb`
**Part 1 with bug-focused comments** - Explains the bugs and fixes.

- ✅ Complete Part 1 implementation
- ✅ Comments around bugs explaining what was incorrect and what is the correct fix
- ✅ Focuses on debugging process and bug explanations

**Use this if**: You want to understand what bugs existed and how they were fixed.

---

#### `SmolLM_DPO_SFT_Part1_Debug_Exercise_Solution_Babysit_20250628.ipynb`
**Part 1 with comprehensive step-by-step implementation guide** - Detailed walkthrough.

- ✅ Complete Part 1 implementation
- ✅✅✅ **VERY comprehensive comments**
- ✅ Explains how the custom SmolLM architecture is implemented step-by-step
- ✅ Documents the entire thought process and implementation logic
- ✅ This notebook is like babysitting you through understanding the entire Part 1 implementation
- ✅ Best for learning the complete implementation process in detail

**Use this if**: You want a detailed, educational walkthrough of how to implement the custom SmolLM architecture from scratch.

---

### 🚀 Part 2 Notebook (SFT/DPO Implementation)

#### `SmolLM_SFT_DPO_Part2.ipynb`
**Part 2 only** - SFT and DPO implementation without Part 1.

- ✅ Complete Part 2 implementation (SFT + DPO + DPO variants)
- ❌ Does NOT include Part 1 (custom architecture implementation)
- ✅ **Start here if you want to skip Part 1 entirely**
- ✅ Combines DPO implementation and DPO variants (like IPO) into one section

**Use this if**: You're only interested in the SFT/DPO implementation and don't care about the custom architecture exercise.

---

## Learning Paths

### Path 1: Complete Learner (Everything)
1. `SmolLM_SFT_DPO_Original.ipynb` - Understand the scope
2. `SmolLM_DPO_SFT_Part1_Debug_Exercise_Solution_Babysit_20250628.ipynb` - Learn Part 1 in detail
3. `SmolLM_SFT_DPO_Full.ipynb` - See the complete implementation and run experiments

### Path 2: Quick Implementation Focus
1. `SmolLM_SFT_DPO_Part2.ipynb` - Skip Part 1, go straight to SFT/DPO
2. `SmolLM_SFT_DPO_Full.ipynb` - See the full project with results

### Path 3: Self-Implementation Challenge
1. `SmolLM_SFT_DPO_Original.ipynb` - Read the questions
2. Implement your own solution
3. Compare with `SmolLM_SFT_DPO_Full.ipynb`

### Path 4: Results-Oriented (Just Show Me!)
1. `SmolLM_SFT_DPO_Full.ipynb` - Run this and get all the results immediately

---

## File Organization

```
notebook/
├── README.md (you are here)
├── SmolLM_SFT_DPO_Original.ipynb                            # Original questions only
├── SmolLM_SFT_DPO_Part1_Reproduce_Raw.ipynb                 # Part 1 minimal comments
├── SmolLM_SFT_DPO_Part1_Debug_Exercise_Solution.ipynb       # Part 1 moderate comments
├── SmolLM_DPO_SFT_Part1_Debug_Exercise_Solution_Babysit_20250628.ipynb  # Part 1 comprehensive
├── SmolLM_SFT_DPO_Part2.ipynb                               # Part 2 only (skip Part 1)
└── SmolLM_SFT_DPO_Full.ipynb                                # MAIN: Full implementation
```

**Note**: Generated files will be saved in the `notebook/` directory, including:
- Model checkpoints (e.g., `./sft-smollm-SFTpadding-model/`)
- Datasets (e.g., `./dpo_preference_dataset/`)
- Plots and results (e.g., `./dpo_current.png`, `bleu_results.json`)

---

## Backup File

**There is a backup notebook at the project root**:
`/home/zhuoyuan/projects/SmolLM-GEC-SFT-DPO/SmolLM_SFT_DPO_Implementation_Backup_DataCollatorForCompletionOnlyLM.ipynb`

- This is a **fallback copy** of `SmolLM_SFT_DPO_Full.ipynb`
- Kept at the root to avoid breaking anything during reorganization
- **Always try running notebooks in the `notebook/` folder first**
- Only use the backup if something goes wrong with the notebooks folder

---

## Expected Results

When you run the full notebook (`SmolLM_SFT_DPO_Full.ipynb`), you should expect:

- **SFT Training**: Multiple training approaches (padding, packing)
- **Preference Dataset Generation**: ~19K preference pairs for DPO
- **DPO/IPO Training**: Direct Preference Optimization
- **BLEU Score Evaluation**: Performance metrics for grammatical error correction
- **Training Curves**: Visualizations of loss and learning rate schedules

Typical BLEU scores:
- Baseline (no training): ~0.05
- SFT: ~0.30-0.49
- DPO/IPO: ~0.49-0.50+

## Expected Directory Structure

When you run `SmolLM_SFT_DPO_Part2.ipynb` or `SmolLM_SFT_DPO_Full.ipynb`, the following files and directories will be generated in the `notebook/` directory:

**Note**: Both Part 2 and Full notebooks will generate the same directory structure since only Part 2 generates output files. Part 1 does not create additional files.

```
notebook/
│
├── # Notebooks (your original files)
├── SmolLM_SFT_DPO_Full.ipynb
├── SmolLM_SFT_DPO_Part2.ipynb
├── ... (other notebooks)
│
├── # SFT Training Results
├── sft-smollm-SFTpadding/              # SFT training checkpoints (padding method)
│   ├── checkpoint-XXX/                 # Training checkpoints
│   └── ...
├── sft-smollm-SFTpadding-model/        # Final SFT model (padding)
│   ├── model.safetensors               # Model weights (~514MB)
│   ├── config.json                     # Model configuration
│   ├── tokenizer.json                  # Tokenizer files
│   ├── vocab.json
│   ├── merges.txt
│   └── ... (other config files)
│
├── sft-smollm-packing/                 # SFT training (dataset packing method)
│   └── checkpoint-XXX/
├── sft-smollm-SFTpacking-model/        # Final SFT model (dataset packing)
│
├── sft-smollm-packing-batch/           # SFT training (batch packing method)
│   └── checkpoint-XXX/
├── sft-smollm-SFTpackingBatch-model/   # Final SFT model (batch packing)
│
├── # DPO Training Results
├── dpo-smollm-model/                   # DPO training checkpoints
│   ├── checkpoint-XXX/
│   └── ...
├── dpo-smollm-final-model/             # Final DPO model
│   ├── model.safetensors
│   ├── config.json
│   └── ... (similar structure to SFT models)
│
├── # IPO Training Results (DPO variant)
├── ipo-smollm-model/                   # IPO training checkpoints
│   └── checkpoint-XXX/
├── ipo-smollm-final-model/             # Final IPO model
│
├── # Preference Dataset for DPO/IPO
├── dpo_preference_dataset/             # Final preference dataset (~70MB)
│   └── cache-*.arrow                   # Dataset cache files
├── dpo_preference_dataset.json         # Preference dataset JSON (~6.7MB)
├── dpo_dataset_stats.json              # Dataset statistics
│
├── dpo_preference_dataset_temp_2000/   # Temporary datasets (saved every 2000 samples)
├── dpo_preference_dataset_temp_4000/
├── dpo_preference_dataset_temp_6000/
├── ... (continues up to 18000)
├── dpo_preference_dataset_temp_18000/
│
├── # Evaluation Results
├── bleu_results.json                   # BLEU scores for all models
│
├── # Training Visualizations
├── dpo_current.png                     # DPO training curves
├── ipo_training_curves.png             # IPO training curves
├── ipo_training_history.json           # IPO training metrics
│
├── # Downloaded Models (HuggingFace cache)
└── C4AI_SMOLLM135/                     # Base SmolLM-135M model
    └── ... (model files)
```

**Storage Requirements:**
- SFT models: ~500-600MB each (3 models = ~1.5-2GB)
- DPO/IPO models: ~500MB each (2 models = ~1GB)
- Preference datasets: ~150-200MB total (including temp files)
- Training checkpoints: ~1-2GB (varies based on save strategy)
- **Total estimated space**: ~5-8GB

**Note**: You can safely delete the `dpo_preference_dataset_temp_*` directories after the preference dataset is fully generated to save space (~50-100MB).

---

If you encounter any issues:
1. Make sure you're running notebooks from the `notebook/` directory
2. Check that all dependencies are installed (see main project README)
3. Ensure HuggingFace cache is properly configured
4. Try the backup notebook if the main ones have issues


