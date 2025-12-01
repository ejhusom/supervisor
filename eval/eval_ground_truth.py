#!/usr/bin/env python3
"""
Generate ground truth for log file evaluation.

Usage:
    python eval_ground_truth.py <log_file> > ground_truth.json
"""

import sys
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


class BGLLogParser:
    """Parser for Blue Gene/L log format."""
    
    @staticmethod
    def parse_line(line):
        """Parse a BGL log line into components."""
        parts = line.strip().split(None, 9)
        if len(parts) < 10:
            return None
        
        try:
            # BGL format: - timestamp timestamp timestamp node_id level component message
            return {
                "timestamp": parts[4],  # Use the main timestamp
                "node_id": parts[5],
                "level": parts[8],
                "component": parts[7],
                "message": parts[9]
            }
        except (IndexError, ValueError):
            return None


def extract_statistics(log_file):
    """Extract verifiable statistics from log file."""
    
    lines = []
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Parse all lines
    parsed_lines = []
    for line in lines:
        parsed = BGLLogParser.parse_line(line)
        if parsed:
            parsed_lines.append(parsed)
    
    # Basic counts
    total_lines = len(parsed_lines)
    level_counts = Counter(p["level"] for p in parsed_lines)
    component_counts = Counter(p["component"] for p in parsed_lines)
    node_counts = Counter(p["node_id"] for p in parsed_lines)
    
    # Temporal analysis - group by hour
    hour_counts = defaultdict(int)
    hour_level_counts = defaultdict(lambda: defaultdict(int))
    
    for p in parsed_lines:
        try:
            # Extract hour from timestamp (format: 2005-06-03-15.42.50.363779)
            timestamp = p["timestamp"]
            hour = timestamp.split('-')[3].split('.')[0]  # Get hour part
            hour_counts[hour] += 1
            hour_level_counts[hour][p["level"]] += 1
        except (IndexError, ValueError):
            pass
    
    # Pattern extraction - find common error messages
    error_messages = [p["message"] for p in parsed_lines if p["level"] == "ERROR"]
    
    # Extract common patterns from error messages
    error_patterns = Counter()
    for msg in error_messages:
        # Extract key phrases (simple word-based patterns)
        words = msg.lower().split()
        if len(words) >= 3:
            # Take first 3-5 significant words as pattern
            pattern = ' '.join(words[:min(5, len(words))])
            error_patterns[pattern] += 1
    
    # Find peak error periods
    peak_hour = max(hour_level_counts.items(), 
                   key=lambda x: x[1].get("ERROR", 0))[0] if hour_level_counts else None
    
    # Time span
    timestamps = [p["timestamp"] for p in parsed_lines]
    time_span = {
        "start": timestamps[0] if timestamps else None,
        "end": timestamps[-1] if timestamps else None
    }
    
    # Unique counts
    unique_nodes_count = len(node_counts)
    unique_components_count = len(component_counts)
    
    # Busiest and quietest hours
    busiest_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
    quietest_hour = min(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None
    
    # Busiest node
    busiest_node = node_counts.most_common(1)[0][0] if node_counts else None
    
    # Hours with errors
    hours_with_errors = [h for h, levels in hour_level_counts.items() if levels.get("ERROR", 0) > 0]
    
    # Component with most errors
    component_error_counts = Counter()
    for p in parsed_lines:
        if p["level"] == "ERROR":
            component_error_counts[p["component"]] += 1
    component_with_most_errors = component_error_counts.most_common(1)[0][0] if component_error_counts else None
    
    # Ratios
    info_count = level_counts.get("INFO", 0)
    warning_count = level_counts.get("WARNING", 0)
    
    return {
        "total_lines": total_lines,
        "level_counts": dict(level_counts),
        "component_counts": dict(component_counts.most_common(10)),  # Top 10
        "node_counts": dict(node_counts.most_common(10)),
        "hour_counts": dict(hour_counts),
        "hour_level_counts": {h: dict(levels) for h, levels in hour_level_counts.items()},
        "error_patterns": dict(error_patterns.most_common(10)),
        "peak_error_hour": peak_hour,
        "time_span": time_span,
        "unique_nodes_count": unique_nodes_count,
        "unique_components_count": unique_components_count,
        "busiest_hour": busiest_hour,
        "quietest_hour": quietest_hour,
        "busiest_node": busiest_node,
        "hours_with_errors": hours_with_errors,
        "component_with_most_errors": component_with_most_errors,
        "component_error_counts": dict(component_error_counts.most_common(10)),
        "info_count": info_count,
        "warning_count": warning_count
    }


def generate_evaluation_questions(stats):
    """Generate evaluation questions from statistics."""
    questions = []
    
    # Q1: Total error count
    error_count = stats["level_counts"].get("ERROR", 0)
    questions.append({
        "id": "q1_error_count",
        "question": "How many ERROR level log entries are in this file?",
        "answer_type": "integer",
        "expected": error_count,
        "tolerance": 0
    })
    
    # Q2: Total lines
    questions.append({
        "id": "q2_total_lines",
        "question": "How many total log lines are in this file?",
        "answer_type": "integer",
        "expected": stats["total_lines"],
        "tolerance": 0
    })
    
    # Q3: Top component
    if stats["component_counts"]:
        top_component = max(stats["component_counts"].items(), key=lambda x: x[1])
        questions.append({
            "id": "q3_top_component",
            "question": "Which component has the most log entries?",
            "answer_type": "string_match",
            "expected": [top_component[0]],
            "case_sensitive": False
        })
    
    # Q4: Top 3 components
    if len(stats["component_counts"]) >= 3:
        top_3 = sorted(stats["component_counts"].items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_names = [c[0] for c in top_3]
        questions.append({
            "id": "q4_top3_components",
            "question": "What are the top 3 components by log volume?",
            "answer_type": "list",
            "expected": top_3_names,
            "order_matters": False  # Accept any order
        })
    
    # Q5: Peak error hour
    if stats["peak_error_hour"]:
        questions.append({
            "id": "q5_peak_error_hour",
            "question": "During which hour did the most errors occur?",
            "answer_type": "string_match",
            "expected": [
                stats["peak_error_hour"],
                f"hour {stats['peak_error_hour']}",
                f"{stats['peak_error_hour']}:00"
            ],
            "case_sensitive": False
        })
    
    # Q6: Warning count
    warning_count = stats["level_counts"].get("WARNING", 0)
    if warning_count > 0:
        questions.append({
            "id": "q6_warning_count",
            "question": "How many WARNING level entries are there?",
            "answer_type": "integer",
            "expected": warning_count,
            "tolerance": 0
        })
    
    # Q7: Level distribution
    questions.append({
        "id": "q7_level_types",
        "question": "What log levels are present in this file?",
        "answer_type": "list",
        "expected": list(stats["level_counts"].keys()),
        "order_matters": False
    })
    
    # Q8: Error ratio
    if stats["total_lines"] > 0:
        error_ratio = (error_count / stats["total_lines"]) * 100
        questions.append({
            "id": "q8_error_ratio",
            "question": "What percentage of log entries are errors? (answer as integer percentage)",
            "answer_type": "integer",
            "expected": int(round(error_ratio)),
            "tolerance": 2  # Allow ±2% tolerance
        })
    
    # Q9: Unique nodes count
    questions.append({
        "id": "q9_unique_nodes",
        "question": "How many unique nodes generated logs?",
        "answer_type": "integer",
        "expected": stats["unique_nodes_count"],
        "tolerance": 0
    })
    
    # Q10: Busiest node
    if stats.get("busiest_node"):
        questions.append({
            "id": "q10_busiest_node",
            "question": "Which node generated the most logs?",
            "answer_type": "string_match",
            "expected": [stats["busiest_node"]],
            "case_sensitive": False
        })

    # Q11: Has fatal logs
    has_fatal = "FATAL" in stats["level_counts"]
    questions.append({
        "id": "q11_has_fatal",
        "question": "Are there any FATAL level logs in this file? (answer yes or no)",
        "answer_type": "string_match",
        "expected": ["yes", "true", "1"] if has_fatal else ["no", "false", "0"],
        "case_sensitive": False
    })
    
    # Q12: Component with most errors
    if stats.get("component_with_most_errors"):
        questions.append({
            "id": "q12_component_most_errors",
            "question": "Which component generated the most ERROR logs?",
            "answer_type": "string_match",
            "expected": [stats["component_with_most_errors"]],
            "case_sensitive": False
        })
    
    # Q13: Info to Error ratio
    info_count = stats.get("info_count", 0)
    if error_count > 0 and info_count > 0:
        ratio = info_count / error_count
        questions.append({
            "id": "q13_info_error_ratio",
            "question": "What is the ratio of INFO logs to ERROR logs? (answer as integer, rounded)",
            "answer_type": "integer",
            "expected": int(round(ratio)),
            "tolerance": 1
        })
    
    # Q14: More frequent level
    warning_count = stats.get("warning_count", 0)
    if warning_count > 0 and info_count > 0:
        # Determine which is more frequent
        more_frequent = "INFO" if info_count > warning_count else "WARNING"
        questions.append({
            "id": "q14_info_vs_warning",
            "question": "Which level appears more frequently: INFO or WARNING?",
            "answer_type": "string_match",
            "expected": [more_frequent, more_frequent.lower()],
            "case_sensitive": False
        })

    # Q15: Quietest hour
    if stats.get("quietest_hour"):
        questions.append({
            "id": "q15_quietest_hour",
            "question": "During which hour were the fewest logs generated?",
            "answer_type": "string_match",
            "expected": [
                stats["quietest_hour"],
                f"hour {stats['quietest_hour']}",
                f"{stats['quietest_hour']}:00"
            ],
            "case_sensitive": False
        })
    
    # Q16: Busiest hour
    if stats.get("busiest_hour"):
        questions.append({
            "id": "q16_busiest_hour",
            "question": "During which hour were the most logs generated?",
            "answer_type": "string_match",
            "expected": [
                stats["busiest_hour"],
                f"hour {stats['busiest_hour']}",
                f"{stats['busiest_hour']}:00"
            ],
            "case_sensitive": False
        })

    # Q17: Are there more unique nodes or unique components?
    if stats.get("unique_nodes_count") and stats.get("unique_components_count"):
        more_diverse = "nodes" if stats["unique_nodes_count"] > stats["unique_components_count"] else "components"
        questions.append({
            "id": "q17_nodes_vs_components",
            "question": "Are there more unique nodes or unique components?",
            "answer_type": "string_match",
            "expected": [more_diverse],
            "case_sensitive": False
        })
    
    # Q18: Hours with errors
    if stats.get("hours_with_errors"):
        hours_with_errors_count = len(stats["hours_with_errors"])
        questions.append({
            "id": "q18_hours_with_errors",
            "question": "How many hours had at least one ERROR log?",
            "answer_type": "integer",
            "expected": hours_with_errors_count,
            "tolerance": 0
        })
    
    return questions


def generate_ground_truth(log_file):
    """Generate complete ground truth JSON."""
    log_path = Path(log_file)
    
    # Extract statistics
    stats = extract_statistics(log_file)
    
    # Generate questions
    questions = generate_evaluation_questions(stats)
    
    # Build ground truth structure
    ground_truth = {
        "metadata": {
            "file": log_path.name,
            "generated_at": datetime.now().isoformat(),
            "generator_version": "1.1"
        },
        "statistics": stats,
        "evaluation_questions": questions
    }
    
    return ground_truth


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_ground_truth.py <log_file>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  python eval_ground_truth.py logs/bgl_2k.log > ground_truth.json", file=sys.stderr)
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        sys.exit(1)
    
    # Generate and output ground truth
    ground_truth = generate_ground_truth(log_file)
    print(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    main()