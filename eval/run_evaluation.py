#!/usr/bin/env python3
"""
Structured evaluation runner for HDFS log analysis pipeline.

Usage:
    python eval/run_evaluation.py [--variant VARIANT] [--output-dir DIR]
"""
import argparse
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_client import LLMClient
from core.agent import Agent
from core.prompt_registry import get_prompt
from core.logger import get_logger


def load_test_cases(path: str = "eval/test_cases_v1.yaml") -> list:
    """Load test cases from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["test_cases"]


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
    
    # Extract predicted label from detection output
    detection_text = detection["content"]
    predicted_label = None
    if '"label": 0' in detection_text or '"label":0' in detection_text:
        predicted_label = 0
    elif '"label": 1' in detection_text or '"label":1' in detection_text:
        predicted_label = 1
    
    return {
        "parsed_logs": parsed["content"],
        "detection_raw": detection_text,
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
    
    test_cases = load_test_cases()
    
    # Create LLM client using config values
    llm = LLMClient(
        provider=config.get("provider"),
        model=config.get("model"),
        api_key=config.get("api_key")
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "metadata": {
            "timestamp": timestamp,
            "prompt_variant": prompt_variant,
            "model": config.get("model"),
            "provider": config.get("provider"),
            "temperature": config.get("temperature")
        },
        "summary": {
            "total": len(test_cases),
            "correct": 0,
            "incorrect": 0,
            "unparseable": 0
        },
        "test_results": []
    }
    
    for tc in test_cases:
        print(f"\nEvaluating: {tc['id']} ({tc['description']})")
        
        # Determine detector variant
        detector_variant = "hdfs_few_shot" if "few_shot" in prompt_variant else "hdfs_zero_shot"
        
        # Start logging session for this test case
        session_id = logger.start_session(
            task=f"Evaluate {tc['id']}: {tc['description']}",
            config={
                "prompt_variant": prompt_variant,
                "model": config.get("model"),
                "provider": config.get("provider"),
                "temperature": config.get("temperature"),
                "test_id": tc["id"],
                "blk_id": tc["blk_id"],
                "system_prompt_parser": get_prompt("log_preprocessor", prompt_variant),
                "system_prompt_detector": get_prompt("log_anomaly_detector", detector_variant),
                "system_prompt_explainer": get_prompt("log_explainer", prompt_variant)
            },
            session_id=f"eval_{tc['id']}_{timestamp}"
        )
        
        result = run_pipeline(tc["logs"], llm, prompt_variant)
        
        # End logging session
        logger.end_session(final_result=f"Label: {result['predicted_label']}")
        
        correct = result["predicted_label"] == tc["expected_label"]
        if result["predicted_label"] is None:
            results["summary"]["unparseable"] += 1
        elif correct:
            results["summary"]["correct"] += 1
        else:
            results["summary"]["incorrect"] += 1
        
        results["test_results"].append({
            "test_id": tc["id"],
            "blk_id": tc["blk_id"],
            "description": tc["description"],
            "expected_label": tc["expected_label"],
            "predicted_label": result["predicted_label"],
            "correct": correct,
            "detection_raw": result["detection_raw"],
            "explanation": result["explanation"]
        })
        
        status = "✅" if correct else ("⚠️" if result["predicted_label"] is None else "❌")
        print(f"  Expected: {tc['expected_label']}, Got: {result['predicted_label']} {status}")
    
    # Calculate accuracy
    parseable = results["summary"]["total"] - results["summary"]["unparseable"]
    if parseable > 0:
        results["summary"]["accuracy"] = results["summary"]["correct"] / parseable
    else:
        results["summary"]["accuracy"] = 0
    
    # Save results
    output_file = output_path / f"eval_{prompt_variant}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Accuracy: {results['summary']['correct']}/{parseable} ({results['summary']['accuracy']:.1%})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run HDFS analysis evaluation")
    parser.add_argument("--variant", type=str, default="few_shot", help="Prompt variant")
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Output directory")
    args = parser.parse_args()
    
    run_evaluation(args.variant, args.output_dir)


if __name__ == "__main__":
    main()
