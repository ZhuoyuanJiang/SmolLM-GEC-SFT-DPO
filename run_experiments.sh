#!/bin/bash

# =============================================================================
# SmolLM GEC Hyperparameter Search Experiment Runner
# =============================================================================
# This script runs comprehensive hyperparameter experiments across 8 GPUs
# 
# Usage:
#   ./run_experiments.sh [phase]
#   
#   phase: all (default), sft, preference, dpo, results
#
# Make sure to:
# 1. Activate your conda environment first: conda activate sft_dpo_env
# 2. Be in the project root directory
# 3. Have 8 GPUs available (CUDA_VISIBLE_DEVICES 0-7)
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPTS_DIR="scripts"
EXPERIMENTS_DIR="experiments"
ARTIFACTS_DIR="artifacts"
MODELS_DIR="models"

# Create necessary directories
mkdir -p "$EXPERIMENTS_DIR" "$ARTIFACTS_DIR" "$MODELS_DIR"

# Function to check if conda environment is activated
check_environment() {
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo "❌ Error: No conda environment activated!"
        echo "Please run: conda activate SmolLM_gec_project"
        exit 1
    fi
    
    # Check if it's the correct environment
    if [[ "$CONDA_DEFAULT_ENV" != "SmolLM_gec_project" ]] && [[ "$CONDA_DEFAULT_ENV" != "sft_dpo_env" ]]; then
        echo "⚠️  Warning: Expected 'SmolLM_gec_project' environment, got '$CONDA_DEFAULT_ENV'"
        echo "Continue anyway? (y/N)"
        read -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
}

# Function to check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ Error: nvidia-smi not found!"
        exit 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -lt 8 ]]; then
        echo "⚠️  Warning: Only $gpu_count GPUs detected (expected 8)"
        echo "Continuing anyway..."
    else
        echo "✅ Found $gpu_count GPUs"
    fi
}

# Function to wait for background jobs
wait_for_jobs() {
    local job_count=$(jobs -r | wc -l)
    if [[ $job_count -gt 0 ]]; then
        echo "⏳ Waiting for $job_count background jobs to complete..."
        wait
        echo "✅ All jobs completed"
    fi
}

# Function to run SFT experiments
run_sft_experiments() {
    echo ""
    echo "🚀 PHASE 1: SFT EXPERIMENTS"
    echo "============================================"
    
    # First batch (8 experiments in parallel)
    echo "Starting first batch of SFT experiments (8 parallel)..."
    
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 8e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=2 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 5e-5 \
        --gradient_accumulation_steps 2 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=3 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 8e-5 \
        --gradient_accumulation_steps 2 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=4 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 5e-5 \
        --gradient_accumulation_steps 4 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=5 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 32 --learning_rate 8e-5 \
        --gradient_accumulation_steps 4 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=6 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 4 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=7 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 4 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    wait_for_jobs
    
    # Second batch (8 experiments in parallel - adding padding bs8 and bs16)
    echo "Starting second batch of SFT experiments (8 parallel)..."
    
    # Packing experiments
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 8 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 8 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=2 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 16 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=3 python "$SCRIPTS_DIR/sft_train.py" \
        --method dataset_packing --batch_size 16 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    # NEW: Padding bs8 and bs16 experiments  
    CUDA_VISIBLE_DEVICES=4 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 8 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=5 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 8 --learning_rate 8e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=6 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 16 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=7 python "$SCRIPTS_DIR/sft_train.py" \
        --method padding --batch_size 16 --learning_rate 8e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    wait_for_jobs
    
    # Third batch: batch_packing experiments (6 experiments)
    echo ""
    echo "Starting third batch: batch_packing experiments (6 parallel)..."
    
    # Smaller batch sizes for batch_packing due to memory efficiency
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 4 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 4 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=2 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 8 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=3 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 8 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=4 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 16 --learning_rate 3e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=5 python "$SCRIPTS_DIR/sft_train.py" \
        --method batch_packing --batch_size 16 --learning_rate 5e-5 \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    wait_for_jobs
    
    echo "✅ All SFT experiments completed (22 total: 16 base + 6 batch_packing)!"
}

# Function to create preference dataset
create_preference_dataset() {
    echo ""
    echo "🔍 PHASE 2: FINDING BEST SFT MODEL & CREATING PREFERENCE DATASET"
    echo "================================================================="
    
    # Find best SFT model
    echo "Finding best SFT model..."
    BEST_SFT=$(python "$SCRIPTS_DIR/aggregate_results.py" \
        --experiments_dir "$EXPERIMENTS_DIR" \
        --phase sft \
        --return_best)
    
    if [[ -z "$BEST_SFT" ]]; then
        echo "❌ Error: Could not find best SFT model!"
        exit 1
    fi
    
    echo "✅ Best SFT model: $BEST_SFT"
    
    # Create preference dataset
    echo "Creating preference dataset..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS_DIR/create_preference_dataset.py" \
        --sft_model_path "$BEST_SFT/final_model" \
        --output_dir "$ARTIFACTS_DIR/preference_dataset" \
        --batch_size 128
    
    echo "✅ Preference dataset created!"
}

# Function to run DPO/IPO experiments
run_dpo_experiments() {
    echo ""
    echo "🎯 PHASE 3: DPO/IPO EXPERIMENTS"
    echo "==============================="
    
    # Find best SFT model (should already exist from previous phase)
    BEST_SFT=$(python "$SCRIPTS_DIR/aggregate_results.py" \
        --experiments_dir "$EXPERIMENTS_DIR" \
        --phase sft \
        --return_best)
    
    if [[ -z "$BEST_SFT" ]]; then
        echo "❌ Error: Could not find best SFT model!"
        exit 1
    fi
    
    echo "Using best SFT model: $BEST_SFT"
    
    # Check if preference dataset exists
    PREFERENCE_DATASET="$ARTIFACTS_DIR/preference_dataset/preference_dataset"
    if [[ ! -d "$PREFERENCE_DATASET" ]]; then
        echo "❌ Error: Preference dataset not found at $PREFERENCE_DATASET"
        echo "Run preference phase first!"
        exit 1
    fi
    
    # Run DPO/IPO experiments (6 total, all in parallel)
    echo "Starting DPO/IPO experiments (6 parallel)..."
    
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method dpo --learning_rate 1e-7 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method dpo --learning_rate 3e-7 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=2 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method dpo --learning_rate 1e-6 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=3 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method ipo --learning_rate 1e-7 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=4 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method ipo --learning_rate 3e-7 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    CUDA_VISIBLE_DEVICES=5 python "$SCRIPTS_DIR/dpo_ipo_train.py" \
        --method ipo --learning_rate 1e-6 \
        --sft_model_path "$BEST_SFT/final_model" \
        --preference_dataset_path "$PREFERENCE_DATASET" \
        --base_output_dir "$EXPERIMENTS_DIR" &
    
    wait_for_jobs
    
    echo "✅ DPO/IPO experiments completed!"
}

# Function to aggregate results
aggregate_results() {
    echo ""
    echo "📊 PHASE 4: AGGREGATING RESULTS"
    echo "==============================="
    
    # Generate comprehensive results
    python "$SCRIPTS_DIR/aggregate_results.py" \
        --experiments_dir "$EXPERIMENTS_DIR" \
        --output_dir "$ARTIFACTS_DIR" \
        --generate_plots \
        --verbose
    
    # Find and save best overall model
    BEST_MODEL=$(python "$SCRIPTS_DIR/aggregate_results.py" \
        --experiments_dir "$EXPERIMENTS_DIR" \
        --return_best)
    
    if [[ -n "$BEST_MODEL" ]]; then
        echo "Copying best model to $MODELS_DIR/best_gec_model..."
        cp -r "$BEST_MODEL/final_model" "$MODELS_DIR/best_gec_model"
        
        # Save best model info
        echo "{\"best_model_path\": \"$BEST_MODEL\", \"timestamp\": \"$(date -Iseconds)\"}" > \
            "$ARTIFACTS_DIR/best_model_info.json"
        
        echo "✅ Best model saved to $MODELS_DIR/best_gec_model"
    fi
    
    echo "✅ Results aggregation completed!"
    echo ""
    echo "📋 SUMMARY FILES CREATED:"
    echo "  - $ARTIFACTS_DIR/experiment_results.csv"
    echo "  - $ARTIFACTS_DIR/summary_statistics.json"
    echo "  - $ARTIFACTS_DIR/bleu_by_method.png"
    echo "  - $ARTIFACTS_DIR/sft_bleu_vs_batch_size.png"
    echo "  - $ARTIFACTS_DIR/learning_rate_analysis.png"
}

# Function to show estimated time
show_time_estimates() {
    echo ""
    echo "⏱️  ESTIMATED TIMES:"
    echo "==================="
    echo "  SFT experiments (22): ~60-75 minutes (parallel)"
    echo "    - 16 base experiments"
    echo "    - 6 batch_packing experiments"
    echo "  Preference dataset:   ~20-30 minutes"
    echo "  DPO/IPO experiments:  ~30-45 minutes (parallel)"
    echo "  Results aggregation:  ~2-5 minutes"
    echo "  TOTAL:               ~2.5-3 hours"
}

# Main function
main() {
    local phase=${1:-all}
    
    echo "🔬 SmolLM GEC Hyperparameter Search"
    echo "==================================="
    echo "Phase: $phase"
    echo "Timestamp: $(date)"
    
    # Pre-flight checks
    check_environment
    check_gpus
    
    case $phase in
        "sft")
            show_time_estimates
            run_sft_experiments
            ;;
        "preference")
            create_preference_dataset
            ;;
        "dpo")
            run_dpo_experiments
            ;;
        "results")
            aggregate_results
            ;;
        "all")
            show_time_estimates
            echo ""
            read -p "Continue with full experiment pipeline? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Experiment cancelled."
                exit 0
            fi
            
            run_sft_experiments
            create_preference_dataset
            run_dpo_experiments
            aggregate_results
            ;;
        *)
            echo "❌ Error: Unknown phase '$phase'"
            echo "Valid phases: all, sft, preference, dpo, results"
            exit 1
            ;;
    esac
    
    echo ""
    echo "🎉 EXPERIMENT PIPELINE COMPLETED!"
    echo "================================="
    echo "Check $ARTIFACTS_DIR/ for results and plots"
    echo "Best model saved to $MODELS_DIR/best_gec_model"
    echo ""
    echo "💡 TIP: Run 'python create_training_configs.py' to generate"
    echo "        training configuration files for all experiments"
}

# Help function
show_help() {
    echo "Usage: $0 [phase]"
    echo ""
    echo "Phases:"
    echo "  all        - Run complete experiment pipeline (default)"
    echo "  sft        - Run only SFT experiments"
    echo "  preference - Create preference dataset (requires SFT results)"
    echo "  dpo        - Run only DPO/IPO experiments (requires preference dataset)"
    echo "  results    - Aggregate and analyze results"
    echo ""
    echo "Examples:"
    echo "  $0           # Run complete pipeline"
    echo "  $0 sft       # Run only SFT experiments"
    echo "  $0 results   # Generate results summary"
}

# Handle help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"