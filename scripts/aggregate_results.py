#!/usr/bin/env python3
"""
Aggregate Results Script for SmolLM GEC Experiments
Collects results from all experiments and generates summary tables
"""

import argparse
import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    
    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Directory containing all experiment results")
    parser.add_argument("--output_dir", type=str, default="artifacts",
                        help="Directory to save aggregated results")
    parser.add_argument("--phase", type=str, choices=["sft", "dpo", "all"], default="all",
                        help="Which phase to analyze")
    parser.add_argument("--return_best", action="store_true",
                        help="Return path to best model (for use in scripts)")
    parser.add_argument("--generate_plots", action="store_true",
                        help="Generate comparison plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def find_experiment_results(experiments_dir: str) -> List[str]:
    """Find all results.json files in experiment directories"""
    pattern = os.path.join(experiments_dir, "*", "results.json")
    results_files = glob.glob(pattern)
    
    if not results_files:
        print(f"No results.json files found in {experiments_dir}")
        return []
    
    # Use stderr for debug messages when return_best is used
    debug_out = sys.stderr if '--return_best' in sys.argv else sys.stdout
    print(f"Found {len(results_files)} experiment results", file=debug_out)
    return results_files


def load_experiment_result(results_path: str) -> Optional[Dict[str, Any]]:
    """Load a single experiment result"""
    try:
        with open(results_path, 'r') as f:
            result = json.load(f)
        
        # Add experiment directory info
        result['experiment_dir'] = os.path.dirname(results_path)
        result['experiment_name'] = os.path.basename(os.path.dirname(results_path))
        
        return result
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return None


def parse_experiment_info(experiment_name: str) -> Dict[str, Any]:
    """Parse experiment information from directory name"""
    info = {
        'base_method': None,
        'batch_size': None,
        'learning_rate': None,
        'epochs': None,
        'final_method': None,
        'final_lr': None
    }
    
    # Handle SFT experiments: sft_padding_bs32_lr5e-5_ep1
    if experiment_name.startswith('sft_'):
        parts = experiment_name.split('_')
        if len(parts) >= 4:
            info['base_method'] = parts[1]  # padding or packing
            info['final_method'] = 'SFT'
            
            for part in parts[2:]:
                if part.startswith('bs'):
                    info['batch_size'] = int(part[2:])
                elif part.startswith('lr'):
                    info['learning_rate'] = float(part[2:].replace('e-', 'e-'))
                elif part.startswith('ep'):
                    info['epochs'] = int(part[2:])
    
    # Handle DPO/IPO experiments: dpo_sft_padding_bs32_lr5e-5_ep1_lr3e-7_ep1
    elif experiment_name.startswith('dpo_') or experiment_name.startswith('ipo_'):
        method = experiment_name.split('_')[0]
        info['final_method'] = method.upper()
        
        # Try to extract information from the name
        parts = experiment_name.split('_')
        for i, part in enumerate(parts):
            if part == 'padding' or part == 'packing':
                info['base_method'] = part
            elif part.startswith('bs') and info['batch_size'] is None:
                info['batch_size'] = int(part[2:])
            elif part.startswith('lr') and info['learning_rate'] is None:
                info['learning_rate'] = float(part[2:].replace('e-', 'e-'))
            elif part.startswith('lr') and info['learning_rate'] is not None and info['final_lr'] is None:
                info['final_lr'] = float(part[2:].replace('e-', 'e-'))
            elif part.startswith('ep') and info['epochs'] is None:
                info['epochs'] = int(part[2:])
    
    return info


def create_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comprehensive results table"""
    rows = []
    
    for result in results:
        if not result:
            continue
        
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        experiment_info = parse_experiment_info(result['experiment_name'])
        
        # Extract information from config if not in experiment name
        if experiment_info['base_method'] is None:
            experiment_info['base_method'] = config.get('method', 'unknown')
        if experiment_info['batch_size'] is None:
            experiment_info['batch_size'] = config.get('batch_size', 'unknown')
        if experiment_info['learning_rate'] is None:
            experiment_info['learning_rate'] = config.get('learning_rate', 'unknown')
        if experiment_info['epochs'] is None:
            experiment_info['epochs'] = config.get('num_epochs', 1)
        if experiment_info['final_method'] is None:
            if config.get('method') in ['dpo', 'ipo']:
                experiment_info['final_method'] = config['method'].upper()
            else:
                experiment_info['final_method'] = 'SFT'
        if experiment_info['final_lr'] is None and experiment_info['final_method'] in ['DPO', 'IPO']:
            experiment_info['final_lr'] = config.get('learning_rate', 'unknown')
        
        row = {
            'Experiment': result['experiment_name'],
            'Base Method': experiment_info['base_method'],
            'Batch Size': experiment_info['batch_size'],
            'SFT LR': experiment_info['learning_rate'] if experiment_info['final_method'] == 'SFT' else experiment_info['learning_rate'],
            'Final Method': experiment_info['final_method'],
            'Final LR': experiment_info['final_lr'] if experiment_info['final_lr'] else '-',
            'Epochs': experiment_info['epochs'],
            'BLEU Score': metrics.get('bleu_score', 0.0),
            'Train Loss': metrics.get('train_loss', 0.0),
            'Eval Loss': metrics.get('eval_loss', 0.0),
            'Training Time (min)': metrics.get('training_time', 0.0) / 60.0,
            'Checkpoint Path': result['experiment_dir']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by BLEU score (descending)
    if not df.empty:
        df = df.sort_values('BLEU Score', ascending=False)
        df = df.reset_index(drop=True)
    
    return df


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics"""
    if df.empty:
        return {}
    
    stats = {
        'total_experiments': len(df),
        'best_bleu': df['BLEU Score'].max(),
        'worst_bleu': df['BLEU Score'].min(),
        'mean_bleu': df['BLEU Score'].mean(),
        'std_bleu': df['BLEU Score'].std(),
        'best_experiment': df.iloc[0]['Experiment'],
        'total_training_time': df['Training Time (min)'].sum(),
        'methods_tested': df['Final Method'].unique().tolist(),
        'base_methods_tested': df['Base Method'].unique().tolist()
    }
    
    # Method-wise statistics
    method_stats = {}
    for method in df['Final Method'].unique():
        method_df = df[df['Final Method'] == method]
        method_stats[method] = {
            'count': len(method_df),
            'best_bleu': method_df['BLEU Score'].max(),
            'mean_bleu': method_df['BLEU Score'].mean(),
            'best_experiment': method_df.iloc[0]['Experiment'] if not method_df.empty else None
        }
    
    stats['method_stats'] = method_stats
    
    return stats


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create comparison plots"""
    if df.empty:
        return
    
    plt.style.use('default')
    
    # 1. BLEU Score by Method
    plt.figure(figsize=(10, 6))
    
    # Group by final method
    methods = df['Final Method'].unique()
    method_scores = []
    method_names = []
    
    for method in methods:
        method_df = df[df['Final Method'] == method]
        method_scores.append(method_df['BLEU Score'].tolist())
        method_names.append(f"{method}\n(n={len(method_df)})")
    
    plt.boxplot(method_scores, labels=method_names)
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Distribution by Method')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bleu_by_method.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. BLEU Score vs Batch Size (for SFT)
    sft_df = df[df['Final Method'] == 'SFT']
    if not sft_df.empty:
        plt.figure(figsize=(10, 6))
        
        for base_method in sft_df['Base Method'].unique():
            method_df = sft_df[sft_df['Base Method'] == base_method]
            plt.scatter(method_df['Batch Size'], method_df['BLEU Score'], 
                       label=f'SFT {base_method.title()}', alpha=0.7, s=100)
        
        plt.xlabel('Batch Size')
        plt.ylabel('BLEU Score')
        plt.title('SFT: BLEU Score vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sft_bleu_vs_batch_size.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Learning Rate Analysis
    plt.figure(figsize=(12, 6))
    
    # SFT Learning Rates
    plt.subplot(1, 2, 1)
    if not sft_df.empty:
        for base_method in sft_df['Base Method'].unique():
            method_df = sft_df[sft_df['Base Method'] == base_method]
            plt.scatter(method_df['SFT LR'], method_df['BLEU Score'], 
                       label=f'SFT {base_method.title()}', alpha=0.7, s=100)
        plt.xlabel('SFT Learning Rate')
        plt.ylabel('BLEU Score')
        plt.title('SFT: BLEU Score vs Learning Rate')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # DPO/IPO Learning Rates
    plt.subplot(1, 2, 2)
    dpo_ipo_df = df[df['Final Method'].isin(['DPO', 'IPO'])]
    if not dpo_ipo_df.empty:
        for method in dpo_ipo_df['Final Method'].unique():
            method_df = dpo_ipo_df[dpo_ipo_df['Final Method'] == method]
            # Convert Final LR to numeric, handle '-' values
            final_lrs = []
            bleu_scores = []
            for _, row in method_df.iterrows():
                if row['Final LR'] != '-' and row['Final LR'] != 'unknown':
                    try:
                        final_lrs.append(float(row['Final LR']))
                        bleu_scores.append(row['BLEU Score'])
                    except (ValueError, TypeError):
                        continue
            
            if final_lrs:
                plt.scatter(final_lrs, bleu_scores, label=method, alpha=0.7, s=100)
        
        plt.xlabel('DPO/IPO Learning Rate')
        plt.ylabel('BLEU Score')
        plt.title('DPO/IPO: BLEU Score vs Learning Rate')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find experiment results
    results_files = find_experiment_results(args.experiments_dir)
    if not results_files:
        print("No experiment results found!")
        return
    
    # Load all results
    # Use stderr for debug output when --return_best is used
    debug_out = sys.stderr if '--return_best' in sys.argv else sys.stdout
    print("Loading experiment results...", file=debug_out)
    results = []
    for results_file in results_files:
        result = load_experiment_result(results_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results loaded!", file=debug_out)
        return
    
    print(f"Loaded {len(results)} experiment results", file=debug_out)
    
    # Filter by phase if specified
    if args.phase != "all":
        if args.phase == "sft":
            results = [r for r in results if not r['experiment_name'].startswith(('dpo_', 'ipo_'))]
        elif args.phase == "dpo":
            results = [r for r in results if r['experiment_name'].startswith(('dpo_', 'ipo_'))]
    
    # Create results table
    df = create_results_table(results)
    
    if df.empty:
        print("No valid results to analyze!")
        return
    
    # Generate summary statistics
    stats = generate_summary_stats(df)
    
    # Print summary
    if args.verbose:
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Best BLEU score: {stats['best_bleu']:.4f}")
        print(f"Mean BLEU score: {stats['mean_bleu']:.4f} ± {stats['std_bleu']:.4f}")
        print(f"Best experiment: {stats['best_experiment']}")
        print(f"Total training time: {stats['total_training_time']:.1f} minutes")
        print(f"Methods tested: {', '.join(stats['methods_tested'])}")
        
        print("\nMethod-wise performance:")
        for method, method_stat in stats['method_stats'].items():
            print(f"  {method}: {method_stat['best_bleu']:.4f} (best), "
                  f"{method_stat['mean_bleu']:.4f} (mean), "
                  f"{method_stat['count']} experiments")
        print("="*60)
    
    # Save results table
    table_path = os.path.join(args.output_dir, "experiment_results.csv")
    df.to_csv(table_path, index=False)
    # Use stderr for status messages when --return_best is used
    debug_out = sys.stderr if args.return_best else sys.stdout
    print(f"✅ Results table saved to {table_path}", file=debug_out)
    
    # Save summary statistics
    stats_path = os.path.join(args.output_dir, "summary_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✅ Summary statistics saved to {stats_path}", file=debug_out)
    
    # Generate plots if requested
    if args.generate_plots:
        create_comparison_plots(df, args.output_dir)
    
    # Return best model path if requested (for use in shell scripts)
    if args.return_best and not df.empty:
        best_path = df.iloc[0]['Checkpoint Path']
        print(best_path)
        return
    
    # Print top 5 results
    print(f"\nTop 5 Results (by BLEU Score):")
    print("-" * 100)
    
    display_columns = ['Final Method', 'Base Method', 'Batch Size', 'SFT LR', 'Final LR', 'BLEU Score']
    top_5 = df[display_columns].head(5)
    
    for idx, row in top_5.iterrows():
        # Convert to strings to handle mixed types
        batch_size = str(row['Batch Size']) if pd.notna(row['Batch Size']) else '-'
        sft_lr = f"{row['SFT LR']:.1e}" if pd.notna(row['SFT LR']) else '-'
        final_lr = str(row['Final LR']) if pd.notna(row['Final LR']) else '-'
        
        print(f"{idx+1:2d}. {row['Final Method']:3s} {row['Base Method']:7s} "
              f"BS={batch_size:>3s} SFT_LR={sft_lr:>8s} "
              f"Final_LR={final_lr:>8s} BLEU={row['BLEU Score']:.4f}")
    
    print("-" * 100)


if __name__ == "__main__":
    main()