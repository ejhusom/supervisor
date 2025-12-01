# iExplain Evaluation Analysis Scripts

Complete suite of Python scripts for analyzing and visualizing LLM multi-agent framework evaluation results, with special focus on model, workflow, and preprocessing comparisons.

## 📊 Scripts Overview

### 1. `analyze_groups.py` - **Focused Grouping Analysis** ⭐ NEW
**Best for:** Clear comparisons between models, workflows, and preprocessing options
- Clean, publication-ready visualizations
- Dedicated plots for each grouping dimension
- Efficiency matrix showing accuracy vs speed trade-offs
- Interaction heatmaps for understanding combined effects
- Automatic insights and recommendations

### 2. `analyze_enhanced.py` - **Modular Enhanced Analysis** ⭐ NEW  
**Best for:** Comprehensive analysis with flexible grouping options
- Automatically detects available grouping columns
- Easily adaptable for new columns added to results
- Multiple visualization types (comparisons, interactions, faceted)
- Statistical summaries for each dimension
- Supports any combination of grouping variables

### 3. `analyze_results.py` - **Comprehensive Analysis**
**Best for:** Detailed exploration of all metrics
- 6 different visualizations covering all aspects
- Per-question performance heatmap
- Time distribution analysis
- Performance by answer type

### 4. `analyze_simple.py` - **Simplified Analysis**
**Best for:** Quick overview and key metrics
- 2 focused visualizations
- Combined performance bars
- Efficiency scatter plot

## 🚀 Quick Start

### Basic usage for grouping analysis:
```bash
# Focused grouping analysis (recommended)
python analyze_groups.py results.csv

# Enhanced modular analysis
python analyze_enhanced.py results.csv

# Save all visualizations
python analyze_groups.py results.csv --save
python analyze_enhanced.py results.csv --save-all
```

### Compare multiple runs:
```bash
# Combine multiple CSV files
python analyze_groups.py run1.csv run2.csv run3.csv --save
```

### Custom output:
```bash
# Specify output format and prefix
python analyze_groups.py results.csv --save --format png --output-prefix myproject

# Save to specific directory
python analyze_enhanced.py results.csv --save-all --output-dir ./analysis/
```

## 📈 Key Features

### Grouping Dimensions Analyzed
- **Model**: Compare different LLM models (gpt-4.1-nano, gpt-4o-mini, gpt-5-nano)
- **Workflow**: Simple vs Evaluator workflow performance
- **Preprocessing**: Impact of preprocessing on accuracy and speed

### Visualization Types

#### From `analyze_groups.py`:
1. **Accuracy by Group** - Bar charts showing accuracy for each dimension
2. **Time Performance** - Horizontal bars for time comparison
3. **Efficiency Matrix** - Scatter plot of accuracy vs speed with zones
4. **Interaction Heatmaps** - Shows combined effects of different variables

#### From `analyze_enhanced.py`:
1. **Comparison Plots** - Side-by-side accuracy and time for each group
2. **Interaction Effects** - Bar plots showing variable interactions
3. **Faceted Analysis** - Multi-dimensional grouped comparisons
4. **Statistical Summaries** - Detailed tables with metrics


## 🔧 Extensibility

The scripts are designed to be easily adaptable for new columns:

### Adding New Grouping Columns
Simply add columns to your CSV, and `analyze_enhanced.py` will automatically detect them:
```python
# Automatically detected columns:
# model, workflow, preprocessing, provider, embeddings, temperature, etc.
```

### Customizing Visualizations
Edit the `COLORS` dictionary in `analyze_groups.py`:
```python
COLORS = {
    'model': {'new-model': '#123456'},
    'new_column': {'value1': '#color1', 'value2': '#color2'}
}
```

## 📁 Output Files

### Naming Convention
- Main comparison: `grouped_analysis_main_YYYYMMDD_HHMMSS.{pdf|png}`
- Interactions: `grouped_analysis_interaction_YYYYMMDD_HHMMSS.{pdf|png}`
- Enhanced plots: `comparison_`, `interaction_`, `faceted_` with timestamps


## 📝 Example Workflow

```bash
# 1. Run initial analysis
python analyze_groups.py results.csv

# 2. Save comprehensive visualizations
python analyze_enhanced.py results.csv --save-all --format png

# 3. Compare multiple experiments
python analyze_groups.py experiment1.csv experiment2.csv --save

# 4. Generate report-ready figures
python analyze_groups.py results.csv --save --format pdf --output-prefix report
```
