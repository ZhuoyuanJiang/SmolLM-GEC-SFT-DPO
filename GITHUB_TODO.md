# GitHub Upload Preparation Plan

## Overview
After all experiments complete, we'll create GitHub-ready directories alongside the existing symlinks.

## Current Structure (KEEP AS IS)
```
SmolLM-GEC-SFT-DPO/
├── experiments -> /tmp5/zhuoyuan/smollm_experiments/experiments  [SYMLINK - Keep]
├── models -> /tmp5/zhuoyuan/smollm_experiments/models            [SYMLINK - Keep]
├── artifacts -> /tmp5/zhuoyuan/smollm_experiments/artifacts      [SYMLINK - Keep]
```

## New Structure to Add (for GitHub)
```
SmolLM-GEC-SFT-DPO/
├── experiments -> ...                [Keep symlink for future experiments]
├── models -> ...                     [Keep symlink for model storage]
├── artifacts -> ...                  [Keep symlink for generating artifacts]
├── github_experiments/               [NEW - Real folder for GitHub]
│   ├── sft_padding_bs32_lr5e-05_ep1/
│   │   ├── results.json             [~500 bytes - BLEU, losses]
│   │   ├── training_args.json       [~1KB - configuration]
│   │   └── final_model/              [~5MB - configs & tokenizer only]
│   │       ├── config.json
│   │       ├── tokenizer_config.json
│   │       ├── tokenizer.json
│   │       ├── vocab.json
│   │       ├── merges.txt
│   │       ├── special_tokens_map.json
│   │       ├── generation_config.json
│   │       └── added_tokens.json
│   ├── sft_packing_bs4_lr3e-05_ep1/
│   │   ├── results.json
│   │   └── final_model/...
│   └── ... (all 18 experiments)
├── github_artifacts/                 [NEW - Real folder for GitHub]
│   ├── experiment_results.csv       [~5KB - comparison table]
│   ├── summary_statistics.json      [~2KB - aggregated stats]
│   ├── bleu_by_method.png          [~50KB - visualization]
│   ├── learning_rate_analysis.png   [~50KB - visualization]
│   └── best_model_info.json        [~500 bytes - which model won]
└── github_models/                    [NEW - Real folder for GitHub]
    └── README.md                     [Documentation only, no actual models]
```

## Step-by-Step Actions

### 1. Create GitHub directories (preserving symlinks)
```bash
mkdir -p github_experiments
mkdir -p github_artifacts  
mkdir -p github_models
```

### 2. Copy experiment results (small files only)
For each experiment in `/tmp5/zhuoyuan/smollm_experiments/experiments/`:
- ✅ Copy `results.json` (metrics, BLEU scores)
- ✅ Copy `training_args.json` or `config.json` (how it was trained)
- ✅ Copy `README.md` if exists
- ✅ Create `final_model/` folder and copy ONLY these files:
  - `config.json` - Model architecture
  - `tokenizer_config.json` - Tokenizer settings  
  - `tokenizer.json` - Tokenization rules
  - `vocab.json` - Vocabulary
  - `merges.txt` - BPE merges
  - `special_tokens_map.json` - Special tokens
  - `generation_config.json` - Generation settings
  - `added_tokens.json` - New tokens
- ❌ Skip `checkpoint*/` folders (hundreds of MB)
- ❌ Skip `model.safetensors` (500MB+ model weights)
- ❌ Skip `*.bin` files (binary formats)

### 3. Copy artifacts (analysis results only)
From `/tmp5/zhuoyuan/smollm_experiments/artifacts/`:
- ✅ Copy all `.csv` files (results tables)
- ✅ Copy all `.json` files (statistics, best model info)
- ✅ Copy all `.png` files (plots)
- ❌ Skip `preference_dataset/` folder (can be regenerated)
- ❌ Skip any `*_dataset/` folders

### 4. Create models documentation
Create `github_models/README.md` explaining:
- Where the actual models are stored (lab server)
- How to access them
- Performance summary (link to artifacts)

### 5. Update .gitignore
Add entries to ignore symlinked directories but include github_* directories:
```gitignore
# Ignore symlinks (they point to /tmp5)
/experiments
/models
/artifacts

# Include GitHub-ready directories
!github_experiments/
!github_artifacts/
!github_models/
```

## Expected Sizes
- `github_experiments/`: ~90MB total (includes tokenizer files, ~5MB per experiment × 18)
- `github_artifacts/`: ~200KB total (CSVs, JSONs, PNGs)
- `github_models/`: ~2KB (just README)
- **Total GitHub addition**: ~90MB ✅ (still very reasonable for GitHub)

## Benefits of This Approach
1. ✅ Symlinks remain for future experiments (use fast /tmp5 disk)
2. ✅ GitHub gets clean, organized results without huge files
3. ✅ Clear separation between "working directories" and "archive directories"
4. ✅ Can easily update GitHub results after new experiments
5. ✅ Home directory quota unaffected (symlinks = ~0 bytes)

## Command to Execute (After All Experiments)
Show this file to Claude and say:
"All experiments are complete. Please execute the GITHUB_TODO.md plan to prepare for GitHub upload."

## Verification Checklist
- [ ] All 12 SFT experiments have results.json
- [ ] Preference dataset was created
- [ ] All 6 DPO/IPO experiments have results.json  
- [ ] Artifacts contain experiment_results.csv
- [ ] Artifacts contain visualization plots
- [ ] Total size of github_* directories < 1MB