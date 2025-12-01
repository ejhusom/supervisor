#!/usr/bin/env python3
"""
Enhanced modular analysis of iExplain evaluation results with flexible grouping.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import sys
from itertools import combinations


class ResultsAnalyzer:
    """Modular analyzer for evaluation results."""
    
    def __init__(self, df):
        self.df = df
        self.identify_grouping_columns()
        
    def identify_grouping_columns(self):
        """Identify available grouping columns in the dataframe."""
        potential_groups = ['model', 'workflow', 'preprocessing', 'provider', 
                           'embeddings', 'config_name']
        self.group_columns = [col for col in potential_groups if col in self.df.columns]
        print(f"Available grouping columns: {self.group_columns}")
    
    def compute_grouped_metrics(self, group_by):
        """Compute metrics grouped by specified column(s)."""
        if isinstance(group_by, str):
            group_by = [group_by]
        
        metrics = self.df.groupby(group_by).agg({
            'correct': ['mean', 'std', 'count'],
            'time_seconds': ['mean', 'std', 'median']
        }).round(3)
        
        # Flatten column names
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns]
        metrics = metrics.rename(columns={
            'correct_mean': 'accuracy',
            'correct_std': 'accuracy_std',
            'correct_count': 'n_samples',
            'time_seconds_mean': 'avg_time',
            'time_seconds_std': 'time_std',
            'time_seconds_median': 'median_time'
        })
        
        return metrics
    
    def create_comparison_plots(self, save_path=None):
        """Create comprehensive comparison plots for all grouping dimensions."""
        # Determine number of group columns to plot
        plot_groups = [col for col in ['model', 'workflow', 'preprocessing'] 
                      if col in self.group_columns]
        
        if not plot_groups:
            print("Warning: No standard grouping columns found")
            return None
        
        # Calculate grid size
        n_plots = len(plot_groups) * 2  # 2 plots per group (accuracy and time)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig = plt.figure(figsize=(14, 5 * n_rows))
        plot_idx = 1
        
        # Create plots for each grouping dimension
        for group_col in plot_groups:
            # Accuracy comparison
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            self._plot_grouped_accuracy(ax, group_col)
            plot_idx += 1
            
            # Time comparison
            ax = plt.subplot(n_rows, n_cols, plot_idx)
            self._plot_grouped_time(ax, group_col)
            plot_idx += 1
        
        plt.suptitle('Grouped Performance Analysis', fontsize=14, y=1.01)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved comparison plots to: {save_path}")
        
        return fig
    
    def _plot_grouped_accuracy(self, ax, group_col):
        """Plot accuracy comparison for a grouping column."""
        metrics = self.compute_grouped_metrics(group_col)
        
        # Sort by accuracy for better readability
        metrics = metrics.sort_values('accuracy', ascending=True)
        
        # Create horizontal bar plot for better label visibility
        bars = ax.barh(range(len(metrics)), metrics['accuracy'] * 100, 
                      xerr=metrics['accuracy_std'] * 100 if 'accuracy_std' in metrics.columns else None,
                      capsize=5, color='steelblue', alpha=0.8)
        
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics.index)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(f'Accuracy by {group_col.replace("_", " ").title()}')
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metrics['accuracy'] * 100)):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', va='center', fontsize=9)
        
        # Add sample size annotations
        for i, (idx, row) in enumerate(metrics.iterrows()):
            ax.text(2, i, f'n={int(row["n_samples"])}', 
                   va='center', fontsize=8, alpha=0.6)
    
    def _plot_grouped_time(self, ax, group_col):
        """Plot time comparison for a grouping column."""
        metrics = self.compute_grouped_metrics(group_col)
        
        # Sort by time for better readability
        metrics = metrics.sort_values('avg_time', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(metrics)), metrics['avg_time'],
                      xerr=metrics['time_std'] if 'time_std' in metrics.columns else None,
                      capsize=5, color='coral', alpha=0.8)
        
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics.index)
        ax.set_xlabel('Average Time (seconds)')
        ax.set_title(f'Time Performance by {group_col.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metrics['avg_time'])):
            ax.text(val + metrics['time_std'].iloc[i] * 0.1 if 'time_std' in metrics.columns else val + 0.5,
                   bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}s', va='center', fontsize=9)
    
    def create_interaction_plots(self, save_path=None):
        """Create interaction plots showing relationships between grouping variables."""
        # Check which interaction plots we can create
        available_pairs = []
        for pair in [('model', 'workflow'), ('model', 'preprocessing'), 
                    ('workflow', 'preprocessing')]:
            if all(col in self.group_columns for col in pair):
                available_pairs.append(pair)
        
        if not available_pairs:
            print("No interaction plots available")
            return None
        
        n_plots = len(available_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for ax, (var1, var2) in zip(axes, available_pairs):
            self._plot_interaction(ax, var1, var2)
        
        plt.suptitle('Interaction Effects Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved interaction plots to: {save_path}")
        
        return fig
    
    def _plot_interaction(self, ax, var1, var2):
        """Plot interaction between two variables."""
        # Compute grouped metrics
        grouped = self.df.groupby([var1, var2])['correct'].mean() * 100
        
        # Pivot for plotting
        pivot = grouped.unstack(fill_value=0)
        
        # Plot lines for each category of var2
        x_pos = np.arange(len(pivot.index))
        width = 0.35
        
        for i, col in enumerate(pivot.columns):
            offset = width * (i - len(pivot.columns)/2 + 0.5)
            ax.bar(x_pos + offset, pivot[col], width, label=f'{var2}={col}', alpha=0.8)
        
        ax.set_xlabel(var1.replace('_', ' ').title())
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{var1.title()} × {var2.title()} Interaction')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
    
    def create_faceted_analysis(self, save_path=None):
        """Create faceted plots for multi-dimensional analysis."""
        # Check if we have the necessary columns
        if 'model' not in self.group_columns or 'workflow' not in self.group_columns:
            print("Faceted analysis requires both 'model' and 'workflow' columns")
            return None
        
        # Create faceted plot
        fig = plt.figure(figsize=(14, 8))
        
        # 1. Accuracy facet grid
        ax1 = plt.subplot(2, 1, 1)
        self._create_grouped_comparison(ax1, 'correct', 'Accuracy (%)', is_accuracy=True)
        
        # 2. Time facet grid
        ax2 = plt.subplot(2, 1, 2)
        self._create_grouped_comparison(ax2, 'time_seconds', 'Time (seconds)', log_scale=True)
        
        plt.suptitle('Faceted Performance Analysis', fontsize=14, y=1.01)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved faceted analysis to: {save_path}")
        
        return fig
    
    def _create_grouped_comparison(self, ax, metric, ylabel, log_scale=False, is_accuracy=False):
        """Create grouped bar plot for comprehensive comparison."""
        # Group by available dimensions
        group_cols = []
        if 'model' in self.group_columns:
            group_cols.append('model')
        if 'workflow' in self.group_columns:
            group_cols.append('workflow')
        if 'preprocessing' in self.group_columns:
            group_cols.append('preprocessing')
        
        if len(group_cols) < 2:
            return
        
        # Create aggregated data
        agg_func = 'mean' if metric == 'correct' else 'median'
        grouped = self.df.groupby(group_cols)[metric].agg(agg_func)
        
        # Convert to percentage if accuracy
        if is_accuracy:
            grouped = grouped * 100
        
        # Create labels
        labels = [' | '.join(map(str, idx)) if isinstance(idx, tuple) else str(idx) 
                 for idx in grouped.index]
        
        # Plot
        bars = ax.bar(range(len(grouped)), grouped.values, alpha=0.7)
        
        # Color by first grouping variable
        if group_cols:
            first_groups = [str(idx[0]) if isinstance(idx, tuple) else str(idx) 
                          for idx in grouped.index]
            unique_first = list(set(first_groups))
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_first)))
            color_map = dict(zip(unique_first, colors))
            
            for bar, group in zip(bars, first_groups):
                bar.set_color(color_map[group])
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by {" × ".join(group_cols)}')
        ax.grid(True, alpha=0.3, axis='y')
        
        if log_scale:
            ax.set_yscale('log')
    
    def generate_statistical_summary(self):
        """Generate statistical summary tables for each grouping dimension."""
        summary = {}
        
        for group_col in self.group_columns:
            if group_col == 'config_name':  # Skip full config name
                continue
                
            metrics = self.compute_grouped_metrics(group_col)
            metrics['accuracy'] = (metrics['accuracy'] * 100).round(1)
            metrics['avg_time'] = metrics['avg_time'].round(2)
            
            summary[group_col] = metrics[['accuracy', 'avg_time', 'n_samples']]
        
        return summary
    
    def print_comparative_analysis(self):
        """Print detailed comparative analysis."""
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)
        
        summaries = self.generate_statistical_summary()
        
        for group_col, summary in summaries.items():
            print(f"\n📊 By {group_col.replace('_', ' ').title()}:")
            print("-" * 50)
            print(summary.to_string())
            
            # Best performer
            best_acc = summary['accuracy'].idxmax()
            fastest = summary['avg_time'].idxmin()
            print(f"\n  Best accuracy: {best_acc} ({summary.loc[best_acc, 'accuracy']}%)")
            print(f"  Fastest: {fastest} ({summary.loc[fastest, 'avg_time']}s)")
        
        # Preprocessing impact if available
        if 'preprocessing' in self.group_columns:
            prep_summary = summaries['preprocessing']
            if True in prep_summary.index and False in prep_summary.index:
                impact = prep_summary.loc[True, 'accuracy'] - prep_summary.loc[False, 'accuracy']
                print(f"\n💡 Preprocessing Impact: {impact:+.1f}% accuracy improvement")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced modular analysis of evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_enhanced.py results.csv
  
  # Save all visualizations
  python analyze_enhanced.py results.csv --save-all
  
  # Save specific plots
  python analyze_enhanced.py results.csv --save-comparison --save-interaction
  
  # Custom output directory
  python analyze_enhanced.py results.csv --save-all --output-dir ./analysis/
        """
    )
    
    parser.add_argument('csv_files', nargs='+', help='CSV result files to analyze')
    parser.add_argument('--save-all', action='store_true', help='Save all visualizations')
    parser.add_argument('--save-comparison', action='store_true', help='Save comparison plots')
    parser.add_argument('--save-interaction', action='store_true', help='Save interaction plots')
    parser.add_argument('--save-faceted', action='store_true', help='Save faceted analysis')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf',
                       help='Output format (default: pdf)')
    parser.add_argument('--output-dir', help='Output directory for saved files')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    try:
        # Load data
        dfs = []
        for file in args.csv_files:
            if not Path(file).exists():
                print(f"Warning: {file} not found, skipping", file=sys.stderr)
                continue
            df = pd.read_csv(file)
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No valid CSV files found")
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        
        # Create analyzer
        analyzer = ResultsAnalyzer(df)
        
        # Print comparative analysis
        analyzer.print_comparative_analysis()
        
        # Setup output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path('.')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualizations
        if args.save_all or args.save_comparison:
            save_path = output_dir / f"comparison_{timestamp}.{args.format}"
            analyzer.create_comparison_plots(save_path)
        
        if args.save_all or args.save_interaction:
            save_path = output_dir / f"interaction_{timestamp}.{args.format}"
            analyzer.create_interaction_plots(save_path)
        
        if args.save_all or args.save_faceted:
            save_path = output_dir / f"faceted_{timestamp}.{args.format}"
            analyzer.create_faceted_analysis(save_path)
        
        # Show plots if requested
        if not args.no_show:
            # Create all plots if none were saved
            if not any([args.save_all, args.save_comparison, 
                       args.save_interaction, args.save_faceted]):
                analyzer.create_comparison_plots()
                analyzer.create_interaction_plots()
                analyzer.create_faceted_analysis()
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
