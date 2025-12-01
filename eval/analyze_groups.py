#!/usr/bin/env python3
"""
Focused grouping analysis with clean visualizations for model, workflow, and preprocessing comparisons.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import sys


class GroupedAnalyzer:
    """Analyzer focused on key grouping dimensions."""
    
    # Define colors for consistent styling
    COLORS = {
        'model': {'gpt-4.1-nano': '#2E86AB', 'gpt-4o-mini': '#F18F01', 'gpt-5-nano': '#C73E1D'},
        'workflow': {'simple': '#4CAF50', 'evaluator': '#9C27B0'},
        'preprocessing': {True: '#00BCD4', False: '#FF5252'}
    }
    
    def __init__(self, df):
        self.df = df
        self.validate_columns()
        
    def validate_columns(self):
        """Check for required columns and add derived ones if needed."""
        self.available_groups = []
        for col in ['model', 'workflow', 'preprocessing']:
            if col in self.df.columns:
                self.available_groups.append(col)
        
        if not self.available_groups:
            raise ValueError("No grouping columns (model, workflow, preprocessing) found in data")
        
        print(f"Analysis will include: {', '.join(self.available_groups)}")
    
    def create_main_comparison(self, save_path=None):
        """Create the main comparison dashboard."""
        n_groups = len(self.available_groups)
        fig = plt.figure(figsize=(16, 10))
        
        # Main grid: 2x3 for up to 6 plots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall accuracy by each dimension
        plot_idx = 0
        for group in self.available_groups:
            ax = fig.add_subplot(gs[0, plot_idx])
            self._plot_accuracy_bars(ax, group)
            plot_idx += 1
        
        # 2. Time performance by each dimension
        plot_idx = 0
        for group in self.available_groups:
            ax = fig.add_subplot(gs[1, plot_idx])
            self._plot_time_bars(ax, group)
            plot_idx += 1
        
        # 3. Combined efficiency plot (spans bottom row)
        ax_efficiency = fig.add_subplot(gs[2, :])
        self._plot_efficiency_matrix(ax_efficiency)
        
        plt.suptitle('Model, Workflow & Preprocessing Comparison', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved main comparison to: {save_path}")
        
        return fig
    
    def _plot_accuracy_bars(self, ax, group_col):
        """Create clean accuracy bar plot."""
        # Compute metrics
        metrics = self.df.groupby(group_col)['correct'].agg(['mean', 'std', 'count'])
        metrics['mean'] = metrics['mean'] * 100
        metrics['std'] = metrics['std'] * 100
        metrics = metrics.sort_values('mean', ascending=False)
        
        # Get colors
        colors = [self.COLORS.get(group_col, {}).get(idx, '#808080') 
                 for idx in metrics.index]
        
        # Plot bars
        bars = ax.bar(range(len(metrics)), metrics['mean'], 
                      yerr=metrics['std'], capsize=5,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([str(x) for x in metrics.index], rotation=0)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'Accuracy by {group_col.title()}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Add value labels
        for bar, (idx, row) in zip(bars, metrics.iterrows()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{row["mean"]:.0f}%', ha='center', va='bottom', fontsize=10)
            # Add sample size
            ax.text(bar.get_x() + bar.get_width()/2., 2,
                   f'n={int(row["count"])}', ha='center', va='bottom', 
                   fontsize=8, color='gray')
    
    def _plot_time_bars(self, ax, group_col):
        """Create clean time performance bar plot."""
        # Compute metrics
        metrics = self.df.groupby(group_col)['time_seconds'].agg(['mean', 'std', 'median'])
        metrics = metrics.sort_values('mean', ascending=True)
        
        # Get colors
        colors = [self.COLORS.get(group_col, {}).get(idx, '#808080') 
                 for idx in metrics.index]
        
        # Plot horizontal bars for better readability
        bars = ax.barh(range(len(metrics)), metrics['mean'],
                      xerr=metrics['std'], capsize=5,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([str(x) for x in metrics.index])
        ax.set_xlabel('Avg Time (seconds)', fontsize=11)
        ax.set_title(f'Time by {group_col.title()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        
        # Add value labels
        for bar, (idx, row) in zip(bars, metrics.iterrows()):
            width = bar.get_width()
            ax.text(width + row['std'] * 0.1, bar.get_y() + bar.get_height()/2,
                   f'{row["mean"]:.1f}s', ha='left', va='center', fontsize=10)
    
    def _plot_efficiency_matrix(self, ax):
        """Create efficiency matrix showing all combinations."""
        # Create all possible combinations
        if 'model' in self.available_groups and 'workflow' in self.available_groups:
            if 'preprocessing' in self.available_groups:
                # 3D comparison
                grouped = self.df.groupby(['model', 'workflow', 'preprocessing']).agg({
                    'correct': 'mean',
                    'time_seconds': 'mean'
                }).reset_index()
                
                # Create efficiency score (accuracy / log(time))
                grouped['efficiency'] = grouped['correct'] * 100 / np.log1p(grouped['time_seconds'])
                
                # Plot scatter with different markers for preprocessing
                for prep_val in grouped['preprocessing'].unique():
                    subset = grouped[grouped['preprocessing'] == prep_val]
                    
                    for model in subset['model'].unique():
                        model_data = subset[subset['model'] == model]
                        marker = 'o' if prep_val else '^'
                        color = self.COLORS['model'].get(model, '#808080')
                        label = f'{model} ({"with" if prep_val else "no"} prep)'
                        
                        ax.scatter(model_data['time_seconds'], 
                                 model_data['correct'] * 100,
                                 s=150, marker=marker, color=color, 
                                 alpha=0.7, edgecolors='black', linewidth=1,
                                 label=label if len(ax.get_legend_handles_labels()[0]) < 6 else "")
                        
                        # Add workflow annotations
                        for _, row in model_data.iterrows():
                            offset = (3, 3) if row['workflow'] == 'simple' else (-3, -3)
                            ax.annotate(row['workflow'][0].upper(), 
                                      (row['time_seconds'], row['correct'] * 100),
                                      xytext=offset, textcoords='offset points',
                                      fontsize=8, fontweight='bold')
            else:
                # 2D comparison
                grouped = self.df.groupby(['model', 'workflow']).agg({
                    'correct': 'mean',
                    'time_seconds': 'mean'
                })
                
                for model in self.df['model'].unique():
                    if model in grouped.index:
                        model_data = grouped.loc[model]
                        color = self.COLORS['model'].get(model, '#808080')
                        
                        ax.scatter(model_data['time_seconds'], 
                                 model_data['correct'] * 100,
                                 s=200, color=color, alpha=0.7,
                                 edgecolors='black', linewidth=1,
                                 label=model)
                        
                        # Add workflow labels
                        for workflow, row in model_data.iterrows():
                            ax.annotate(workflow, 
                                      (row['time_seconds'], row['correct'] * 100),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=9)
        
        # Add ideal zones
        ax.axhspan(70, 100, alpha=0.1, color='green', label='High Accuracy Zone')
        ax.axvspan(0, 20, alpha=0.1, color='blue', label='Fast Zone')
        
        ax.set_xlabel('Average Time (seconds)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Efficiency Matrix: Accuracy vs Speed Trade-off', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_xlim(left=-2)
        ax.set_ylim(-5, 105)
    
    def create_interaction_heatmap(self, save_path=None):
        """Create interaction heatmaps."""
        available_pairs = []
        for pair in [('model', 'workflow'), ('model', 'preprocessing'), 
                    ('workflow', 'preprocessing')]:
            if all(col in self.available_groups for col in pair):
                available_pairs.append(pair)
        
        if not available_pairs:
            print("No interaction pairs available")
            return None
        
        n_pairs = len(available_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        
        for ax, (var1, var2) in zip(axes, available_pairs):
            # Compute interaction matrix
            pivot = self.df.groupby([var1, var2])['correct'].mean() * 100
            pivot = pivot.unstack(fill_value=0)
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                       vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'},
                       linewidths=1, linecolor='gray')
            
            ax.set_title(f'{var1.title()} × {var2.title()} Interaction',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(var2.title(), fontsize=11)
            ax.set_ylabel(var1.title(), fontsize=11)
        
        plt.suptitle('Interaction Effects on Accuracy', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved interaction heatmap to: {save_path}")
        
        return fig
    
    def print_insights(self):
        """Print key insights and recommendations."""
        print("\n" + "="*60)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Model insights
        if 'model' in self.available_groups:
            model_perf = self.df.groupby('model')['correct'].mean() * 100
            best_model = model_perf.idxmax()
            print(f"\n🤖 MODELS:")
            print(f"  Best: {best_model} ({model_perf[best_model]:.0f}% accuracy)")
            
            # With preprocessing
            if 'preprocessing' in self.available_groups:
                model_prep = self.df[self.df['preprocessing'] == True].groupby('model')['correct'].mean() * 100
                best_with_prep = model_prep.idxmax()
                print(f"  Best with preprocessing: {best_with_prep} ({model_prep[best_with_prep]:.0f}%)")
        
        # Workflow insights
        if 'workflow' in self.available_groups:
            workflow_perf = self.df.groupby('workflow')['correct'].mean() * 100
            best_workflow = workflow_perf.idxmax()
            print(f"\n📋 WORKFLOWS:")
            print(f"  Best: {best_workflow} ({workflow_perf[best_workflow]:.0f}% accuracy)")
            
            workflow_time = self.df.groupby('workflow')['time_seconds'].mean()
            print(f"  Time difference: {abs(workflow_time.iloc[0] - workflow_time.iloc[1]):.1f}s")
        
        # Preprocessing insights
        if 'preprocessing' in self.available_groups:
            prep_impact = self.df.groupby('preprocessing')['correct'].mean() * 100
            improvement = prep_impact[True] - prep_impact[False]
            
            time_impact = self.df.groupby('preprocessing')['time_seconds'].mean()
            time_saving = time_impact[False] - time_impact[True]
            
            print(f"\n⚡ PREPROCESSING:")
            print(f"  Accuracy gain: +{improvement:.0f}%")
            print(f"  Time saved: {time_saving:.1f}s per question")
            print(f"  Recommendation: Always use preprocessing")
        
        # Best configuration
        if len(self.available_groups) >= 2:
            config_cols = [col for col in self.available_groups if col != 'embeddings']
            best_configs = self.df.groupby(config_cols)['correct'].mean().nlargest(3)
            
            print(f"\n🏆 TOP CONFIGURATIONS:")
            for i, (config, acc) in enumerate(best_configs.items(), 1):
                config_str = ' + '.join([f'{k}={v}' for k, v in 
                                        (zip(config_cols, config) if len(config_cols) > 1 else [(config_cols[0], config)])])
                print(f"  {i}. {config_str}: {acc*100:.0f}%")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Focused grouping analysis for model evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_groups.py results.csv
  
  # Save visualizations
  python analyze_groups.py results.csv --save
  
  # Custom output
  python analyze_groups.py results.csv --save --output my_analysis.pdf
        """
    )
    
    parser.add_argument('csv_files', nargs='+', help='CSV result files')
    parser.add_argument('--save', action='store_true', help='Save visualizations')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf')
    parser.add_argument('--output-prefix', help='Output filename prefix')
    parser.add_argument('--no-show', action='store_true', help='Do not display')
    
    args = parser.parse_args()
    
    try:
        # Load data
        dfs = []
        for file in args.csv_files:
            if not Path(file).exists():
                print(f"Warning: {file} not found, skipping", file=sys.stderr)
                continue
            dfs.append(pd.read_csv(file))
        
        if not dfs:
            raise ValueError("No valid CSV files found")
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        # Create analyzer
        analyzer = GroupedAnalyzer(df)
        
        # Print insights
        analyzer.print_insights()
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = args.output_prefix or "grouped_analysis"
        
        if args.save:
            # Main comparison
            save_path = f"{prefix}_main_{timestamp}.{args.format}"
            analyzer.create_main_comparison(save_path)
            
            # Interaction heatmap
            save_path = f"{prefix}_interaction_{timestamp}.{args.format}"
            analyzer.create_interaction_heatmap(save_path)
        else:
            # Just create for display
            analyzer.create_main_comparison()
            analyzer.create_interaction_heatmap()
        
        if not args.no_show:
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
