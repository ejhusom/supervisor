#!/usr/bin/env python3
"""
Structured evaluation runner for HDFS log analysis pipeline.

Data format (matching iExplain-logAnalysis):
  - eval/ground_truth.csv: BlockId,Label (Anomaly/Normal)
  - eval/sessions/*.log: One log file per block session

Usage:
    # Run with experiment config
    uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json

    # Run all configs in a file
    uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json --all

    # Run single config by name
    uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json --name baseline_3agent_gpt4o_mini_few_shot

    # Legacy: Run with prompt variant directly
    uv run python3 eval/run_evaluation.py --variant few_shot
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_client import LLMClient
from core.agent import Agent
from core.prompt_registry import get_prompt
from core.logger import get_logger


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    task: str
    provider: str
    model: str
    pipeline: str
    description: str = ""
    prompts: dict = None
    workflow: str = "simple"
    tools_available: list = None
    tools_unavailable: list = None
    agents_available: list = None
    max_iterations: int = 10
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(
            name=data["name"],
            task=data.get("task", "anomaly_detection"),
            provider=data["provider"],
            model=data["model"],
            pipeline=data["pipeline"],
            description=data.get("description", ""),
            prompts=data.get("prompts", {}),
            workflow=data.get("workflow", "simple"),
            tools_available=data.get("tools_available"),
            tools_unavailable=data.get("tools_unavailable"),
            agents_available=data.get("agents_available"),
            max_iterations=data.get("max_iterations", 10),
        )
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "pipeline": self.pipeline,
            "description": self.description,
            "prompts": self.prompts,
            "workflow": self.workflow,
            "tools_available": self.tools_available,
            "tools_unavailable": self.tools_unavailable,
            "agents_available": self.agents_available,
            "max_iterations": self.max_iterations,
        }


def load_experiment_configs(config_path: str) -> list[ExperimentConfig]:
    """Load experiment configurations from JSON file."""
    with open(config_path) as f:
        data = json.load(f)
    
    # Handle both single config and array of configs
    if isinstance(data, list):
        return [ExperimentConfig.from_dict(cfg) for cfg in data]
    else:
        return [ExperimentConfig.from_dict(data)]


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


def run_hardcoded_3agent_pipeline(
    session_logs: str, 
    llm_client: LLMClient, 
    prompts: dict
) -> dict:
    """Run hardcoded 3-agent pipeline (Parser → Detector → Explainer)."""
    
    parser_variant = prompts.get("parser", "few_shot")
    detector_variant = prompts.get("detector", "hdfs_few_shot")
    explainer_variant = prompts.get("explainer", "few_shot")
    
    # Agent 1: Parser
    parser = Agent(
        name="log_parser",
        system_prompt=get_prompt("log_preprocessor", parser_variant),
        llm_client=llm_client,
        tools={},
        logging_enabled=True
    )
    parsed = parser.run(f"Parse these logs:\n{session_logs}")
    
    # Agent 2: Detector
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
        system_prompt=get_prompt("log_explainer", explainer_variant),
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


def run_supervisor_pipeline(
    session_logs: str,
    exp_config: ExperimentConfig,
    llm_client: LLMClient
) -> dict:
    """Run supervisor-based pipeline with configurable workflow."""
    from core.supervisor import Supervisor
    from registry.tool_registry import ToolRegistry
    from registry.agent_registry import AgentRegistry
    from workflows import SimpleWorkflow, EvaluatorWorkflow
    
    # Create supervisor with config restrictions
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry()
    
    supervisor = Supervisor(
        llm_client=llm_client,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        instructions_dir="instructions",
        tools_available=exp_config.tools_available,
        tools_unavailable=exp_config.tools_unavailable,
        agents_available=exp_config.agents_available,
    )
    
    # Create workflow
    if exp_config.workflow == "evaluator":
        workflow = EvaluatorWorkflow(supervisor, max_iterations=3, verbose=False)
    else:
        workflow = SimpleWorkflow(supervisor)
    
    # Construct task prompt
    task = f"""Analyze the following HDFS log session for anomalies.

1. Parse the logs to extract message bodies
2. Detect if this is a normal (0) or anomalous (1) session
3. Explain your reasoning

Output a JSON object with: {{"label": 0 or 1, "signals": ["observation1", "observation2"]}}

Log session:
{session_logs}
"""
    
    # Run workflow
    result = workflow.run(task)
    
    # Extract label from result
    result_text = result.get("content", "") if isinstance(result, dict) else str(result)
    predicted_label = normalize_label(result_text)
    
    return {
        "parsed_logs": "",  # Not separately tracked in supervisor mode
        "detection_raw": result_text,
        "predicted_label": predicted_label,
        "explanation": result_text
    }


def run_pipeline_for_config(
    session_logs: str,
    exp_config: ExperimentConfig,
    llm_client: LLMClient
) -> dict:
    """Dispatch to appropriate pipeline based on config."""
    
    if exp_config.pipeline == "hardcoded_3agent":
        return run_hardcoded_3agent_pipeline(
            session_logs, 
            llm_client, 
            exp_config.prompts or {}
        )
    elif exp_config.pipeline == "supervisor":
        return run_supervisor_pipeline(
            session_logs,
            exp_config,
            llm_client
        )
    else:
        raise ValueError(f"Unknown pipeline type: {exp_config.pipeline}")


def run_experiment(exp_config: ExperimentConfig, output_dir: str = "eval/results") -> dict:
    """Run evaluation for a single experiment configuration."""
    
    # Configure logger for evaluation
    logger = get_logger()
    logger.log_dir = Path("eval/logs")
    logger.log_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
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
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_config.name}")
    print(f"{'='*60}")
    print(f"Pipeline: {exp_config.pipeline}")
    print(f"Model: {exp_config.provider}/{exp_config.model}")
    print(f"Test cases: {len(test_cases)}")
    print()
    
    # Get API key from environment
    api_key = None
    if exp_config.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    elif exp_config.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Create LLM client
    llm = LLMClient(
        provider=exp_config.provider,
        model=exp_config.model,
        api_key=api_key
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect predictions
    y_true = []
    y_pred = []
    test_results = []
    unparseable_count = 0
    
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['block_id']}", end=" ")
        
        # Start logging session
        session_id = logger.start_session(
            task=f"Evaluate {tc['block_id']}",
            config={
                "experiment": exp_config.name,
                "pipeline": exp_config.pipeline,
                "model": exp_config.model,
                "provider": exp_config.provider,
                "block_id": tc["block_id"],
            },
            session_id=f"eval_{exp_config.name}_{tc['block_id']}_{timestamp}"
        )
        
        # Run pipeline
        result = run_pipeline_for_config(tc["logs"], exp_config, llm)
        
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
        print(f"expected={expected} got={predicted} {status}")
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["unparseable"] = unparseable_count
    metrics["total"] = len(test_cases)
    
    results = {
        "metadata": {
            "timestamp": timestamp,
            "experiment": exp_config.name,
            "pipeline": exp_config.pipeline,
            "provider": exp_config.provider,
            "model": exp_config.model,
            "prompts": exp_config.prompts,
            "config": exp_config.to_dict()
        },
        "metrics": metrics,
        "test_results": test_results
    }
    
    # Save JSON results
    json_file = output_path / f"eval_{exp_config.name}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_file = output_path / f"eval_{exp_config.name}_{timestamp}_summary.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Model", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "Accuracy"])
        writer.writerow([
            exp_config.name, exp_config.model,
            metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"],
            metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]
        ])
    
    # Print summary
    print(f"\n=== {exp_config.name} Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1:        {metrics['f1']:.2%}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
    print(f"Results: {json_file}")
    
    return results


def run_evaluation(prompt_variant: str, output_dir: str):
    """Legacy function: Run evaluation with prompt variant (backward compatible)."""
    
    # Create a config from legacy parameters
    exp_config = ExperimentConfig(
        name=f"legacy_{prompt_variant}",
        task="anomaly_detection",
        provider=config.get("provider"),
        model=config.get("model"),
        pipeline="hardcoded_3agent",
        prompts={
            "parser": prompt_variant,
            "detector": f"hdfs_{prompt_variant}",
            "explainer": prompt_variant
        }
    )
    
    return run_experiment(exp_config, output_dir)


def run_all_experiments(config_path: str, output_dir: str, filter_name: str = None):
    """Run all experiments from a config file."""
    configs = load_experiment_configs(config_path)
    
    if filter_name:
        configs = [c for c in configs if filter_name in c.name]
    
    if not configs:
        print(f"No experiments found matching filter: {filter_name}")
        return []
    
    print(f"Running {len(configs)} experiment(s)...")
    
    all_results = []
    for i, exp_config in enumerate(configs, 1):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i}/{len(configs)}")
        print(f"{'#'*60}")
        
        result = run_experiment(exp_config, output_dir)
        all_results.append(result)
    
    # Print combined summary
    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")
    for r in all_results:
        m = r["metrics"]
        name = r["metadata"]["experiment"]
        print(f"{name}: Acc={m['accuracy']:.2%} P={m['precision']:.2%} R={m['recall']:.2%} F1={m['f1']:.2%}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run HDFS log analysis evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with experiment config (first config in file)
  uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json

  # Run all configs in file
  uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json --all

  # Run specific config by name
  uv run python3 eval/run_evaluation.py --experiment experiments/anomaly_detection/baseline_3agent.json --name gpt4o_mini

  # Legacy: Run with prompt variant
  uv run python3 eval/run_evaluation.py --variant few_shot
        """
    )
    parser.add_argument("--experiment", "-e", type=str, help="Experiment config JSON file")
    parser.add_argument("--all", "-a", action="store_true", help="Run all configs in file")
    parser.add_argument("--name", "-n", type=str, help="Filter configs by name (substring match)")
    parser.add_argument("--variant", type=str, help="Legacy: Prompt variant (few_shot/zero_shot)")
    parser.add_argument("--output-dir", "-o", type=str, default="eval/results", help="Output directory")
    args = parser.parse_args()
    
    if args.experiment:
        # New experiment-based mode
        if args.all or args.name:
            run_all_experiments(args.experiment, args.output_dir, args.name)
        else:
            # Run first config only
            configs = load_experiment_configs(args.experiment)
            if configs:
                run_experiment(configs[0], args.output_dir)
            else:
                print(f"No configs found in {args.experiment}")
    elif args.variant:
        # Legacy mode
        run_evaluation(args.variant, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Specify either --experiment or --variant")
        sys.exit(1)


if __name__ == "__main__":
    main()
