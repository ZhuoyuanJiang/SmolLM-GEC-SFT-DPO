#!/bin/bash

# Script to copy experiment results to GitHub-ready directories
# This preserves only essential files, skipping large model weights

set -e

echo "📦 Preparing GitHub-ready experiment data..."
echo "==========================================="

# Source and destination directories (use symlinks for portability)
# Resolve symlinks to get actual paths
SRC_EXPERIMENTS=$(readlink -f "experiments")
SRC_ARTIFACTS=$(readlink -f "artifacts")
DST_EXPERIMENTS="github_experiments"
DST_ARTIFACTS="github_artifacts"

echo "Using source directories:"
echo "  Experiments: $SRC_EXPERIMENTS"
echo "  Artifacts: $SRC_ARTIFACTS"
echo ""

# Function to copy experiment data
copy_experiment() {
    local exp_name=$1
    local src_dir="$SRC_EXPERIMENTS/$exp_name"
    local dst_dir="$DST_EXPERIMENTS/$exp_name"
    
    if [ ! -d "$src_dir" ]; then
        echo "⚠️  Warning: $exp_name not found, skipping..."
        return
    fi
    
    echo "📁 Processing $exp_name..."
    mkdir -p "$dst_dir"
    
    # Copy essential files
    [ -f "$src_dir/results.json" ] && cp "$src_dir/results.json" "$dst_dir/"
    [ -f "$src_dir/training_config.json" ] && cp "$src_dir/training_config.json" "$dst_dir/"
    [ -f "$src_dir/training_history.json" ] && cp "$src_dir/training_history.json" "$dst_dir/"
    [ -f "$src_dir/README.md" ] && cp "$src_dir/README.md" "$dst_dir/"
    
    # Copy tokenizer config files only (no model weights)
    if [ -d "$src_dir/final_model" ]; then
        mkdir -p "$dst_dir/final_model"
        
        # Copy configuration and tokenizer files
        for file in config.json tokenizer_config.json tokenizer.json vocab.json \
                   merges.txt special_tokens_map.json generation_config.json \
                   added_tokens.json; do
            [ -f "$src_dir/final_model/$file" ] && cp "$src_dir/final_model/$file" "$dst_dir/final_model/"
        done
    fi
}

# Process all experiments
echo ""
echo "📊 Copying experiment results..."
echo "--------------------------------"

# SFT experiments (22 total: 10 padding + 6 dataset_packing + 6 batch_packing)
# Note: Also handles old "sft_packing_*" naming for backward compatibility

# Padding experiments (10)
for exp in sft_padding_bs8_lr5e-05_ep1 sft_padding_bs8_lr8e-05_ep1 \
           sft_padding_bs16_lr5e-05_ep1 sft_padding_bs16_lr8e-05_ep1 \
           sft_padding_bs32_lr5e-05_ep1 sft_padding_bs32_lr8e-05_ep1 \
           sft_padding_bs64_lr5e-05_ep1 sft_padding_bs64_lr8e-05_ep1 \
           sft_padding_bs128_lr5e-05_ep1 sft_padding_bs128_lr8e-05_ep1; do
    copy_experiment "$exp"
done

# Dataset packing experiments (6) - try both old and new naming
for exp in sft_dataset_packing_bs4_lr3e-05_ep1 sft_dataset_packing_bs4_lr5e-05_ep1 \
           sft_dataset_packing_bs8_lr3e-05_ep1 sft_dataset_packing_bs8_lr5e-05_ep1 \
           sft_dataset_packing_bs16_lr3e-05_ep1 sft_dataset_packing_bs16_lr5e-05_ep1 \
           sft_packing_bs4_lr3e-05_ep1 sft_packing_bs4_lr5e-05_ep1 \
           sft_packing_bs8_lr3e-05_ep1 sft_packing_bs8_lr5e-05_ep1 \
           sft_packing_bs16_lr3e-05_ep1 sft_packing_bs16_lr5e-05_ep1; do
    copy_experiment "$exp"
done

# Batch packing experiments (6) - new method
for exp in sft_batch_packing_bs4_lr3e-05_ep1 sft_batch_packing_bs4_lr5e-05_ep1 \
           sft_batch_packing_bs8_lr3e-05_ep1 sft_batch_packing_bs8_lr5e-05_ep1 \
           sft_batch_packing_bs16_lr3e-05_ep1 sft_batch_packing_bs16_lr5e-05_ep1; do
    copy_experiment "$exp"
done

# DPO/IPO experiments
for exp in dpo_final_model_lr1e-07_ep1 dpo_final_model_lr3e-07_ep1 dpo_final_model_lr1e-06_ep1 \
           ipo_final_model_lr1e-07_ep1 ipo_final_model_lr3e-07_ep1 ipo_final_model_lr1e-06_ep1; do
    copy_experiment "$exp"
done

echo ""
echo "📈 Copying artifacts..."
echo "----------------------"

# Copy main artifact files
cp "$SRC_ARTIFACTS/experiment_results.csv" "$DST_ARTIFACTS/"
cp "$SRC_ARTIFACTS/summary_statistics.json" "$DST_ARTIFACTS/"
cp "$SRC_ARTIFACTS/all_experiments_config.json" "$DST_ARTIFACTS/"
cp "$SRC_ARTIFACTS/best_model_info.json" "$DST_ARTIFACTS/"

# Copy plots
cp "$SRC_ARTIFACTS/"*.png "$DST_ARTIFACTS/" 2>/dev/null || echo "No plots found"

# Copy preference dataset (JSON files only)
echo ""
echo "📝 Copying preference dataset..."
echo "--------------------------------"
mkdir -p "$DST_ARTIFACTS/preference_dataset"
cp "$SRC_ARTIFACTS/preference_dataset/preference_dataset.json" "$DST_ARTIFACTS/preference_dataset/"
cp "$SRC_ARTIFACTS/preference_dataset/preference_dataset_human_readable.json" "$DST_ARTIFACTS/preference_dataset/"
cp "$SRC_ARTIFACTS/preference_dataset/preference_dataset_sample.json" "$DST_ARTIFACTS/preference_dataset/"
cp "$SRC_ARTIFACTS/preference_dataset/results.json" "$DST_ARTIFACTS/preference_dataset/"

echo ""
echo "📊 Size check..."
echo "---------------"
du -sh github_experiments/
du -sh github_artifacts/
du -sh github_models/

echo ""
echo "✅ GitHub preparation complete!"
echo "Total size: $(du -sh github_* | awk '{sum+=$1} END {print sum}')MB (approximately)"