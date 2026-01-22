#!/usr/bin/env python3
"""
Generate test cases from iExplain-logAnalysis data.

Outputs:
  - eval/ground_truth.csv: BlockId,Label format (matching iExplain)
  - eval/sessions/*.log: One log file per block session

Usage:
    uv run python3 scripts/generate_test_cases.py [--sample N] [--all]
"""
import argparse
import csv
import os
import random
import shutil

# Paths
IEXPLAIN_DATA_DIR = "data"
LABELS_FILE = os.path.join(IEXPLAIN_DATA_DIR, "HDFS_anomaly_label_385_sampled_balanced.csv")
SESSIONS_DIR = os.path.join(IEXPLAIN_DATA_DIR, "HDFS_385_balanced_sampled_sessions")

OUTPUT_DIR = "eval"
OUTPUT_GT = os.path.join(OUTPUT_DIR, "ground_truth.csv")
OUTPUT_SESSIONS = os.path.join(OUTPUT_DIR, "sessions")


def load_labels(labels_path: str) -> dict:
    """Load ground truth labels from CSV."""
    labels = {}
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['BlockId']] = row['Label']
    return labels


def generate_test_cases(sample_size: int = None, seed: int = 42):
    """Generate test cases from iExplain data."""
    
    # Load all labels
    all_labels = load_labels(LABELS_FILE)
    print(f"Loaded {len(all_labels)} labels from {LABELS_FILE}")
    
    # Separate by label
    normal_blks = [b for b, l in all_labels.items() if l == 'Normal']
    anomaly_blks = [b for b, l in all_labels.items() if l == 'Anomaly']
    print(f"  Normal: {len(normal_blks)}, Anomaly: {len(anomaly_blks)}")
    
    # Select blocks
    if sample_size:
        random.seed(seed)
        n_each = sample_size // 2
        selected_normal = random.sample(normal_blks, min(n_each, len(normal_blks)))
        selected_anomaly = random.sample(anomaly_blks, min(n_each, len(anomaly_blks)))
        selected = selected_normal + selected_anomaly
        print(f"Sampled {len(selected)} blocks ({len(selected_normal)} normal, {len(selected_anomaly)} anomaly)")
    else:
        selected = list(all_labels.keys())
        print(f"Using all {len(selected)} blocks")
    
    # Create output directories
    os.makedirs(OUTPUT_SESSIONS, exist_ok=True)
    
    # Write ground truth CSV
    with open(OUTPUT_GT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['BlockId', 'Label'])
        for blk_id in selected:
            writer.writerow([blk_id, all_labels[blk_id]])
    print(f"Written ground truth to {OUTPUT_GT}")
    
    # Copy log files
    copied = 0
    for blk_id in selected:
        src = os.path.join(SESSIONS_DIR, f"{blk_id}.log")
        dst = os.path.join(OUTPUT_SESSIONS, f"{blk_id}.log")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Warning: Log file not found for {blk_id}")
    
    print(f"Copied {copied} log files to {OUTPUT_SESSIONS}")


def main():
    parser = argparse.ArgumentParser(description="Generate test cases from iExplain data")
    parser.add_argument("--sample", type=int, default=30, 
                        help="Number of test cases to sample (balanced). Use --all for all data.")
    parser.add_argument("--all", action="store_true", help="Use all available data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    sample_size = None if args.all else args.sample
    generate_test_cases(sample_size=sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
