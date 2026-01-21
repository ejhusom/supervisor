#!/usr/bin/env python3
"""
Analyze evaluation results and generate comparison report.

Usage:
    python eval/analyze_results.py [--results-dir DIR] [--output FORMAT]
"""
import argparse
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
        "## Summary by Prompt Variant",
        "",
        "| Variant | Model | Accuracy | Correct | Incorrect | Unparseable | Timestamp |",
        "|---------|-------|----------|---------|-----------|-------------|-----------|"
    ]
    
    for r in results:
        meta = r["metadata"]
        summ = r["summary"]
        accuracy_pct = f"{summ.get('accuracy', 0):.1%}"
        lines.append(
            f"| {meta['prompt_variant']} | {meta['model']} | {accuracy_pct} | "
            f"{summ['correct']} | {summ['incorrect']} | {summ['unparseable']} | "
            f"{meta['timestamp']} |"
        )
    
    lines.extend([
        "",
        "## Per-Test Breakdown",
        ""
    ])
    
    # Collect all test IDs
    all_test_ids = set()
    for r in results:
        for tr in r["test_results"]:
            all_test_ids.add(tr["test_id"])
    
    # Create per-test comparison
    lines.append("| Test ID | Expected | " + " | ".join(r["metadata"]["prompt_variant"] for r in results) + " |")
    lines.append("|---------|----------|" + "|".join(["----------"] * len(results)) + "|")
    
    for test_id in sorted(all_test_ids):
        row = [test_id]
        expected = None
        
        for r in results:
            for tr in r["test_results"]:
                if tr["test_id"] == test_id:
                    if expected is None:
                        expected = tr["expected_label"]
                    predicted = tr["predicted_label"]
                    status = "✅" if tr["correct"] else ("⚠️" if predicted is None else "❌")
                    row.append(f"{predicted} {status}")
                    break
            else:
                row.append("-")
        
        row.insert(1, str(expected))
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def generate_best_config_summary(results: list) -> str:
    """Identify and document the best performing configuration."""
    
    if not results:
        return "No results to analyze."
    
    # Find best by accuracy
    best = max(results, key=lambda r: r["summary"].get("accuracy", 0))
    meta = best["metadata"]
    summ = best["summary"]
    
    lines = [
        "",
        "## Best Performing Configuration",
        "",
        f"**Prompt Variant:** {meta['prompt_variant']}",
        f"**Model:** {meta['model']} ({meta['provider']})",
        f"**Accuracy:** {summ.get('accuracy', 0):.1%} ({summ['correct']}/{summ['total'] - summ['unparseable']})",
        "",
        "### Recommended Settings",
        "",
        "```toml",
        f'provider = "{meta["provider"]}"',
        f'model = "{meta["model"]}"',
        f'temperature = {meta["temperature"]}',
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
    report += generate_best_config_summary(results)
    
    with open(args.output, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {args.output}")
    print("\n" + report)


if __name__ == "__main__":
    main()
