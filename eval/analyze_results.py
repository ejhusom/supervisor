#!/usr/bin/env python3
"""
Analysis and visualization of iExplain evaluation results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import sys


def load_results(csv_files):
    """Load and combine results from multiple CSV files."""
    dfs = []
    for file in csv_files:
        if not Path(file).exists():
            print(f"Warning: {file} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(file)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def compute_metrics(df):
    """Compute aggregate metrics for each configuration."""
    metrics = df.groupby('config_name').agg({
        'correct': ['sum', 'count', 'mean'],
        'time_seconds': ['mean', 'std', 'sum']
    }).round(3)
    
    metrics.columns = ['_'.join(col) for col in metrics.columns]
    metrics = metrics.rename(columns={
        'correct_sum': 'correct_count',
        'correct_count': 'total_questions',
        'correct_mean': 'accuracy',
        'time_seconds_mean': 'avg_time',
        'time_seconds_std': 'time_std',
        'time_seconds_sum': 'total_time'
    })
    
    return metrics


def create_visualizations(df, save_path=None):
    """Create comprehensive visualizations."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Accuracy comparison
    ax1 = plt.subplot(2, 3, 1)
    metrics = compute_metrics(df)
    bars = ax1.bar(range(len(metrics)), metrics['accuracy'] * 100)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics.index, rotation=45, ha='right')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics['accuracy'] * 100):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # 2. Time performance
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(range(len(metrics)), metrics['avg_time'], 
            yerr=metrics['time_std'], capsize=5, alpha=0.7)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics.index, rotation=45, ha='right')
    ax2.set_ylabel('Avg Time per Question (s)')
    ax2.set_title('Time Performance')
    
    # 3. Accuracy vs Time trade-off
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(metrics['avg_time'], metrics['accuracy'] * 100, s=100)
    for idx, name in enumerate(metrics.index):
        ax3.annotate(name, 
                    (metrics['avg_time'].iloc[idx], 
                     metrics['accuracy'].iloc[idx] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Avg Time per Question (s)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy vs Speed Trade-off')
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-question performance heatmap
    ax4 = plt.subplot(2, 3, 4)
    pivot = df.pivot_table(values='correct', index='question_id', 
                           columns='config_name', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax4, cbar_kws={'label': 'Success Rate'})
    ax4.set_title('Per-Question Performance')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Question ID')
    
    # 5. Time distribution boxplot
    ax5 = plt.subplot(2, 3, 5)
    df_plot = df[['config_name', 'time_seconds']].copy()
    df_plot['config_name'] = pd.Categorical(df_plot['config_name'])
    ax5.boxplot([df[df['config_name'] == config]['time_seconds'].values 
                 for config in metrics.index], labels=metrics.index)
    ax5.set_xticklabels(metrics.index, rotation=45, ha='right')
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Time Distribution')
    ax5.set_yscale('log')
    
    # 6. Success by question type
    ax6 = plt.subplot(2, 3, 6)
    type_performance = df.groupby(['config_name', 'answer_type'])['correct'].mean()
    type_df = type_performance.unstack(fill_value=0)
    type_df.plot(kind='bar', ax=ax6)
    ax6.set_xlabel('Configuration')
    ax6.set_ylabel('Success Rate')
    ax6.set_title('Performance by Answer Type')
    ax6.legend(title='Answer Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('iExplain Evaluation Results Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    return fig


def print_summary(df):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    metrics = compute_metrics(df)
    
    # Overall best performer
    best_acc = metrics['accuracy'].idxmax()
    fastest = metrics['avg_time'].idxmin()
    
    print(f"\n📊 Best Accuracy: {best_acc} ({metrics.loc[best_acc, 'accuracy']*100:.1f}%)")
    print(f"⚡ Fastest: {fastest} ({metrics.loc[fastest, 'avg_time']:.2f}s avg)")
    
    # Detailed metrics table
    print("\n" + "-"*70)
    print("Detailed Metrics:")
    print("-"*70)
    
    display_metrics = metrics[['accuracy', 'avg_time', 'total_time']].copy()
    display_metrics['accuracy'] = (display_metrics['accuracy'] * 100).round(1).astype(str) + '%'
    display_metrics['avg_time'] = display_metrics['avg_time'].round(2).astype(str) + 's'
    display_metrics['total_time'] = display_metrics['total_time'].round(1).astype(str) + 's'
    
    print(display_metrics.to_string())
    
    # Question-level insights
    print("\n" + "-"*70)
    print("Question Difficulty (by success rate):")
    print("-"*70)
    
    question_difficulty = df.groupby('question_id')['correct'].mean().sort_values()
    
    print("\nHardest questions:")
    for qid, success_rate in question_difficulty.head(3).items():
        q_text = df[df['question_id'] == qid]['question'].iloc[0][:50] + "..."
        print(f"  {qid}: {success_rate*100:.0f}% - {q_text}")
    
    print("\nEasiest questions:")
    for qid, success_rate in question_difficulty.tail(3).items():
        q_text = df[df['question_id'] == qid]['question'].iloc[0][:50] + "..."
        print(f"  {qid}: {success_rate*100:.0f}% - {q_text}")
    
    # Preprocessing impact
    if 'preprocessing' in df.columns:
        print("\n" + "-"*70)
        print("Preprocessing Impact:")
        print("-"*70)
        
        prep_impact = df.groupby('preprocessing')['correct'].mean()
        print(f"  With preprocessing: {prep_impact.get(True, 0)*100:.1f}%")
        print(f"  Without preprocessing: {prep_impact.get(False, 0)*100:.1f}%")
        
        if True in prep_impact.index and False in prep_impact.index:
            improvement = (prep_impact[True] - prep_impact[False]) * 100
            print(f"  Improvement: {improvement:+.1f}%")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize iExplain evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python analyze_results.py results.csv
  
  # Compare multiple runs
  python analyze_results.py results1.csv results2.csv
  
  # Save visualization
  python analyze_results.py results.csv --save
  python analyze_results.py results.csv --save --format png
        """
    )
    
    parser.add_argument('csv_files', nargs='+', help='CSV result files to analyze')
    parser.add_argument('--save', action='store_true', help='Save visualization to file')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf',
                       help='Output format (default: pdf)')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_results(args.csv_files)
        
        # Print summary
        print_summary(df)
        
        # Create visualizations
        save_path = None
        if args.save:
            if args.output:
                save_path = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"analysis_{timestamp}.{args.format}"
        
        fig = create_visualizations(df, save_path)
        
        if not args.no_show:
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
