#!/usr/bin/env python3
"""
Analyze evaluation results and generate comparison report.

Usage:
    uv run python3 eval/analyze_results.py [--results-dir DIR] [--output FILE]
"""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime


def load_all_results(results_dir: str = "eval/results") -> list:
    """Load all evaluation result files."""
    results_path = Path(results_dir)
    results = []
    
    for f in sorted(results_path.glob("eval_*.json")):
        with open(f) as fp:
            data = json.load(fp)
            data["_filename"] = f.name
            results.append(data)
    
    return results


def generate_comparison_table(results: list) -> str:
    """Generate markdown comparison table."""
    
    lines = [
        "# Evaluation Results Comparison",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary by Configuration",
        "",
        "| Variant | Model | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |",
        "|---------|-------|----------|-----------|--------|----|----|----|----|-----|"
    ]
    
    for r in results:
        meta = r["metadata"]
        # Support both old "summary" and new "metrics" format
        m = r.get("metrics", r.get("summary", {}))
        
        accuracy = m.get("accuracy", 0)
        precision = m.get("precision", 0)
        recall = m.get("recall", 0)
        f1 = m.get("f1", 0)
        
        lines.append(
            f"| {meta['prompt_variant']} | {meta['model']} | "
            f"{accuracy:.2%} | {precision:.2%} | {recall:.2%} | {f1:.2%} | "
            f"{m.get('TP', '-')} | {m.get('FP', '-')} | {m.get('TN', '-')} | {m.get('FN', '-')} |"
        )
    
    return "\n".join(lines)


def generate_per_test_breakdown(results: list) -> str:
    """Generate per-test comparison table."""
    if not results:
        return ""
    
    lines = [
        "",
        "## Per-Test Breakdown",
        ""
    ]
    
    # Collect all block IDs
    all_block_ids = set()
    for r in results:
        for tr in r.get("test_results", []):
            all_block_ids.add(tr.get("block_id", tr.get("blk_id", "unknown")))
    
    if not all_block_ids:
        return ""
    
    # Create header
    header = "| Block ID | Expected | " + " | ".join(r["metadata"]["prompt_variant"] for r in results) + " |"
    separator = "|----------|----------|" + "|".join(["----------"] * len(results)) + "|"
    lines.extend([header, separator])
    
    for block_id in sorted(all_block_ids):
        row = [block_id[:20]]  # Truncate long block IDs
        expected = None
        
        for r in results:
            for tr in r.get("test_results", []):
                tr_id = tr.get("block_id", tr.get("blk_id"))
                if tr_id == block_id:
                    if expected is None:
                        expected = tr["expected_label"]
                    predicted = tr["predicted_label"]
                    status = "✅" if tr["correct"] else ("⚠️" if predicted is None else "❌")
                    row.append(f"{predicted} {status}")
                    break
            else:
                row.append("-")
        
        row.insert(1, str(expected) if expected is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def generate_best_config_summary(results: list) -> str:
    """Identify and document the best performing configuration."""
    
    if not results:
        return "\nNo results to analyze."
    
    # Find best by F1 (more robust than accuracy for imbalanced data)
    def get_f1(r):
        m = r.get("metrics", r.get("summary", {}))
        return m.get("f1", m.get("accuracy", 0))
    
    best = max(results, key=get_f1)
    meta = best["metadata"]
    m = best.get("metrics", best.get("summary", {}))
    
    lines = [
        "",
        "## Best Performing Configuration",
        "",
        f"**Prompt Variant:** {meta['prompt_variant']}",
        f"**Model:** {meta['model']} ({meta['provider']})",
        f"**F1 Score:** {m.get('f1', 0):.2%}",
        f"**Accuracy:** {m.get('accuracy', 0):.2%}",
        "",
        "### Recommended Settings",
        "",
        "```toml",
        f'provider = "{meta["provider"]}"',
        f'model = "{meta["model"]}"',
        f'temperature = {meta.get("temperature", 0.0)}',
        "",
        "[prompts]",
        f'log_preprocessor = "{meta["prompt_variant"]}"',
        f'log_anomaly_detector = "hdfs_{meta["prompt_variant"]}"',
        f'log_explainer = "{meta["prompt_variant"]}"',
        "```",
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results-dir", type=str, default="eval/results")
    parser.add_argument("--output", type=str, default="eval/RESULTS_SUMMARY.md")
    args = parser.parse_args()
    
    results = load_all_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(results)} evaluation result(s)")
    
    report = generate_comparison_table(results)
    report += generate_per_test_breakdown(results)
    report += generate_best_config_summary(results)
    
    with open(args.output, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {args.output}")
    print("\n" + report)


if __name__ == "__main__":
    main()
