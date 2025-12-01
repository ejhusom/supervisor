#!/usr/bin/env python3
"""
Simplified analysis of iExplain evaluation results - focuses on key metrics.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
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


def create_key_visualizations(df, save_path=None):
    """Create simplified, focused visualizations."""
    # Set clean style
    sns.set_style("white")
    plt.rcParams['figure.dpi'] = 100
    
    # Compute metrics
    metrics = df.groupby('config_name').agg({
        'correct': 'mean',
        'time_seconds': 'mean'
    }).round(3)
    metrics.columns = ['accuracy', 'avg_time']
    metrics['accuracy'] *= 100  # Convert to percentage
    
    # Create figure with 2 key plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Combined bar chart - Accuracy and Time
    ax1 = axes[0]
    x = range(len(metrics))
    width = 0.35
    
    # Accuracy bars
    bars1 = ax1.bar([i - width/2 for i in x], metrics['accuracy'], 
                    width, label='Accuracy (%)', color='#2E86AB')
    
    # Time bars on secondary y-axis
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar([i + width/2 for i in x], metrics['avg_time'], 
                         width, label='Avg Time (s)', color='#A23B72')
    
    # Formatting
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Accuracy (%)', color='#2E86AB')
    ax1_twin.set_ylabel('Avg Time (s)', color='#A23B72')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics.index, rotation=45, ha='right')
    ax1.set_title('Performance Comparison')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 105)
    
    # Add value labels
    for bar, val in zip(bars1, metrics['accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Efficiency plot - Accuracy vs Speed trade-off
    ax2 = axes[1]
    
    # Color by model type
    colors = []
    for name in metrics.index:
        if 'gpt4_1_nano' in name:
            colors.append('#2E86AB')
        elif 'gpt4o_mini' in name:
            colors.append('#F18F01')
        else:  # gpt5_nano
            colors.append('#C73E1D')
    
    scatter = ax2.scatter(metrics['avg_time'], metrics['accuracy'], 
                         s=150, c=colors, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for idx, name in enumerate(metrics.index):
        # Simplify names for display
        display_name = name.replace('baseline_', '').replace('evaluator_', 'eval_')
        display_name = display_name.replace('_with_preprocessing', '+prep')
        display_name = display_name.replace('_no_preprocessing', '')
        
        ax2.annotate(display_name, 
                    (metrics['avg_time'].iloc[idx], metrics['accuracy'].iloc[idx]),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Average Time per Question (seconds)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Efficiency Analysis')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(left=-5)
    ax2.set_ylim(-5, 105)
    
    # Add quadrant lines for reference
    median_time = metrics['avg_time'].median()
    median_acc = metrics['accuracy'].median()
    ax2.axvline(median_time, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(median_acc, color='gray', linestyle='--', alpha=0.3)
    
    # Add legend for models
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='GPT-4.1-nano'),
        Patch(facecolor='#F18F01', label='GPT-4o-mini'),
        Patch(facecolor='#C73E1D', label='GPT-5-nano')
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.suptitle('iExplain Evaluation Results', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    return fig


def print_insights(df):
    """Print key insights from the data."""
    metrics = df.groupby('config_name').agg({
        'correct': 'mean',
        'time_seconds': 'mean'
    })
    metrics.columns = ['accuracy', 'avg_time']
    
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    
    # Best overall
    best_acc = metrics['accuracy'].idxmax()
    print(f"\n✅ Best Accuracy: {best_acc.replace('_', ' ')}")
    print(f"   → {metrics.loc[best_acc, 'accuracy']*100:.0f}% correct")
    
    # Fastest
    fastest = metrics['avg_time'].idxmin()
    print(f"\n⚡ Fastest: {fastest.replace('_', ' ')}")
    print(f"   → {metrics.loc[fastest, 'avg_time']:.1f}s per question")
    
    # Preprocessing impact
    with_prep = df[df['preprocessing'] == True]['correct'].mean()
    without_prep = df[df['preprocessing'] == False]['correct'].mean()
    print(f"\n📊 Preprocessing Impact:")
    print(f"   → With: {with_prep*100:.0f}% | Without: {without_prep*100:.0f}%")
    print(f"   → Improvement: {(with_prep - without_prep)*100:+.0f}%")
    
    # Model comparison
    print(f"\n🤖 Model Performance (with preprocessing):")
    model_perf = df[df['preprocessing'] == True].groupby('model')['correct'].mean()
    for model, acc in model_perf.sort_values(ascending=False).items():
        print(f"   → {model}: {acc*100:.0f}%")
    
    print("\n" + "="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Simplified analysis of iExplain evaluation results'
    )
    
    parser.add_argument('csv_files', nargs='+', help='CSV result files')
    parser.add_argument('--save', action='store_true', help='Save visualization')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--no-show', action='store_true', help='Do not display')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_results(args.csv_files)
        
        # Print insights
        print_insights(df)
        
        # Create visualizations
        save_path = None
        if args.save:
            if args.output:
                save_path = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"analysis_simple_{timestamp}.{args.format}"
        
        fig = create_key_visualizations(df, save_path)
        
        if not args.no_show:
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
