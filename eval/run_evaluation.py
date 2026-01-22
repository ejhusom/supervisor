#!/usr/bin/env python3
"""
Structured evaluation runner for HDFS log analysis pipeline.

Data format (matching iExplain-logAnalysis):
  - eval/ground_truth.csv: BlockId,Label (Anomaly/Normal)
  - eval/sessions/*.log: One log file per block session

Usage:
    uv run python3 eval/run_evaluation.py [--variant VARIANT] [--output-dir DIR]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_client import LLMClient
from core.agent import Agent
from core.prompt_registry import get_prompt
from core.logger import get_logger


def load_ground_truth(path: str = "eval/ground_truth.csv") -> dict:
    """Load ground truth labels from CSV (iExplain format)."""
    gt = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert Anomaly/Normal to 1/0
            label = 1 if row['Label'].strip().lower() == 'anomaly' else 0
            gt[row['BlockId']] = label
    return gt


def load_sessions(sessions_dir: str = "eval/sessions") -> dict:
    """Load log sessions from directory."""
    sessions = {}
    sessions_path = Path(sessions_dir)
    for log_file in sessions_path.glob("*.log"):
        block_id = log_file.stem
        sessions[block_id] = log_file.read_text().strip()
    return sessions


def normalize_label(text: str) -> int | None:
    """
    Extract predicted label from LLM output.
    Matches iExplain's normalize_log_analysis_result() logic.
    """
    if text is None:
        return None
    
    text = str(text).strip()
    
    # Try to find JSON-style label first
    if '"label": 0' in text or '"label":0' in text:
        return 0
    if '"label": 1' in text or '"label":1' in text:
        return 1
    
    # Remove long numeric sequences (timestamps, block IDs)
    cleaned = re.sub(r"\d{4,}", " ", text)
    cleaned = re.sub(r"[^\d]", " ", cleaned)
    
    # Find standalone 0 or 1
    matches = re.findall(r"\b[01]\b", cleaned)
    if matches:
        return int(matches[-1])
    
    # Fallback: keyword matching
    if re.search(r"normal", text, re.I):
        return 0
    if re.search(r"anomal", text, re.I):
        return 1
    
    return None


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute classification metrics (matching iExplain's evaluate_and_save_log_analysis).
    """
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    total = len(y_true)
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def run_pipeline(session_logs: str, llm_client: LLMClient, prompt_variant: str):
    """Run 3-agent pipeline, return structured result."""
    
    # Agent 1: Parser
    parser = Agent(
        name="log_parser",
        system_prompt=get_prompt("log_preprocessor", prompt_variant),
        llm_client=llm_client,
        tools={},
        logging_enabled=True
    )
    parsed = parser.run(f"Parse these logs:\n{session_logs}")
    
    # Agent 2: Detector
    detector_variant = "hdfs_few_shot" if "few_shot" in prompt_variant else "hdfs_zero_shot"
    detector = Agent(
        name="anomaly_detector", 
        system_prompt=get_prompt("log_anomaly_detector", detector_variant),
        llm_client=llm_client,
        tools={},
        logging_enabled=True
    )
    detection = detector.run(f"Analyze:\n{parsed['content']}")
    
    # Agent 3: Explainer
    explainer = Agent(
        name="explainer",
        system_prompt=get_prompt("log_explainer", prompt_variant),
        llm_client=llm_client,
        tools={},
        logging_enabled=True
    )
    explanation = explainer.run(
        f"Logs:\n{parsed['content']}\n\nDetection:\n{detection['content']}"
    )
    
    # Extract predicted label using normalize function
    predicted_label = normalize_label(detection["content"])
    
    return {
        "parsed_logs": parsed["content"],
        "detection_raw": detection["content"],
        "predicted_label": predicted_label,
        "explanation": explanation["content"]
    }


def run_evaluation(prompt_variant: str, output_dir: str):
    """Run full evaluation and save results."""
    
    # Configure logger for evaluation
    logger = get_logger()
    logger.log_dir = Path("eval/logs")
    logger.log_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data (new format)
    ground_truth = load_ground_truth()
    sessions = load_sessions()
    
    # Match sessions with ground truth
    test_cases = []
    for block_id, logs in sessions.items():
        if block_id in ground_truth:
            test_cases.append({
                "block_id": block_id,
                "logs": logs,
                "expected_label": ground_truth[block_id]
            })
    
    print(f"Loaded {len(test_cases)} test cases")
    
    # Create LLM client using config values
    llm = LLMClient(
        provider=config.get("provider"),
        model=config.get("model"),
        api_key=config.get("api_key")
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect predictions
    y_true = []
    y_pred = []
    test_results = []
    unparseable_count = 0
    
    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] Evaluating: {tc['block_id']}")
        
        # Determine detector variant
        detector_variant = "hdfs_few_shot" if "few_shot" in prompt_variant else "hdfs_zero_shot"
        
        # Start logging session
        session_id = logger.start_session(
            task=f"Evaluate {tc['block_id']}",
            config={
                "prompt_variant": prompt_variant,
                "model": config.get("model"),
                "provider": config.get("provider"),
                "temperature": config.get("temperature"),
                "block_id": tc["block_id"],
                "system_prompt_parser": get_prompt("log_preprocessor", prompt_variant),
                "system_prompt_detector": get_prompt("log_anomaly_detector", detector_variant),
                "system_prompt_explainer": get_prompt("log_explainer", prompt_variant)
            },
            session_id=f"eval_{tc['block_id']}_{timestamp}"
        )
        
        result = run_pipeline(tc["logs"], llm, prompt_variant)
        
        # End logging session
        logger.end_session(final_result=f"Label: {result['predicted_label']}")
        
        # Collect for metrics
        expected = tc["expected_label"]
        predicted = result["predicted_label"]
        
        if predicted is not None:
            y_true.append(expected)
            y_pred.append(predicted)
        else:
            unparseable_count += 1
        
        correct = predicted == expected if predicted is not None else False
        
        test_results.append({
            "block_id": tc["block_id"],
            "expected_label": expected,
            "predicted_label": predicted,
            "correct": correct,
            "detection_raw": result["detection_raw"],
            "explanation": result["explanation"]
        })
        
        status = "✅" if correct else ("⚠️" if predicted is None else "❌")
        print(f"  Expected: {expected}, Got: {predicted} {status}")
    
    # Compute metrics (matching iExplain format)
    metrics = compute_metrics(y_true, y_pred)
    metrics["unparseable"] = unparseable_count
    metrics["total"] = len(test_cases)
    
    results = {
        "metadata": {
            "timestamp": timestamp,
            "prompt_variant": prompt_variant,
            "model": config.get("model"),
            "provider": config.get("provider"),
            "temperature": config.get("temperature")
        },
        "metrics": metrics,
        "test_results": test_results
    }
    
    # Save JSON results (detailed)
    json_file = output_path / f"eval_{prompt_variant}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary (iExplain compatible)
    csv_file = output_path / f"eval_{prompt_variant}_{timestamp}_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "Accuracy"])
        writer.writerow([
            metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"],
            metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]
        ])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results saved to: {json_file}")
    print(f"Summary saved to: {csv_file}")
    print(f"\n=== Evaluation Summary ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1:        {metrics['f1']:.2%}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
    if unparseable_count > 0:
        print(f"Unparseable: {unparseable_count}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run HDFS analysis evaluation")
    parser.add_argument("--variant", type=str, default="few_shot", help="Prompt variant")
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Output directory")
    args = parser.parse_args()
    
    run_evaluation(args.variant, args.output_dir)


if __name__ == "__main__":
    main()
