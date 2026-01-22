#!/usr/bin/env python3
"""
Demo: HDFS Log Analysis Pipeline

Usage:
    python examples/demo_hdfs_analysis.py [--sessions N] [--prompt-variant VARIANT]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_client import LLMClient
from core.agent import Agent
from core.prompt_registry import get_prompt
from tools.hdfs import extract_hdfs_sessions


def run_pipeline(session_logs: str, llm_client: LLMClient, prompt_variant: str = "default"):
    """Run the 3-agent pipeline on a single session."""
    
    # Agent 1: Parser
    parser_prompt = get_prompt("log_preprocessor", prompt_variant)
    parser = Agent(
        name="log_parser",
        system_prompt=parser_prompt,
        llm_client=llm_client,
        tools={}
    )
    parsed_result = parser.run(f"Parse these logs:\n{session_logs}")
    parsed_logs = parsed_result["content"]
    
    # Agent 2: Anomaly Detector
    # Map prompt variant to detector-specific variant
    if prompt_variant == "few_shot":
        detector_variant = "hdfs_few_shot"
    elif prompt_variant == "zero_shot":
        detector_variant = "hdfs_zero_shot"
    else:
        detector_variant = "default"
    
    detector_prompt = get_prompt("log_anomaly_detector", detector_variant)
    detector = Agent(
        name="anomaly_detector",
        system_prompt=detector_prompt,
        llm_client=llm_client,
        tools={}
    )
    detection_result = detector.run(f"Analyze these parsed logs:\n{parsed_logs}")
    
    # Agent 3: Explainer
    explainer_prompt = get_prompt("log_explainer", prompt_variant)
    explainer = Agent(
        name="explainer",
        system_prompt=explainer_prompt,
        llm_client=llm_client,
        tools={}
    )
    explanation = explainer.run(
        f"Parsed logs:\n{parsed_logs}\n\nDetection result:\n{detection_result['content']}"
    )
    
    return {
        "parsed": parsed_logs,
        "detection": detection_result["content"],
        "explanation": explanation["content"]
    }


def main():
    parser = argparse.ArgumentParser(description="HDFS Log Analysis Demo")
    parser.add_argument("--sessions", type=int, default=3, help="Number of sessions to analyze")
    parser.add_argument("--prompt-variant", type=str, default="few_shot", help="Prompt variant")
    parser.add_argument("--log-file", type=str, default="workspace/data/HDFS_2k.log")
    args = parser.parse_args()
    
    # Load logs
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    with open(log_path) as f:
        content = f.read()
    
    # Extract sessions
    sessions = extract_hdfs_sessions(content, max_sessions=args.sessions)
    print(f"Extracted {len(sessions)} sessions")
    
    # Initialize LLM
    llm = LLMClient(
        provider=config["provider"],
        model=config["model"],
        api_key=config.get("api_key")
    )
    
    # Process each session
    for blk_id, lines in sessions.items():
        print(f"\n{'='*60}")
        print(f"Session: {blk_id} ({len(lines)} lines)")
        print('='*60)
        
        session_text = '\n'.join(lines)
        result = run_pipeline(session_text, llm, args.prompt_variant)
        
        print("\n--- EXPLANATION ---")
        print(result["explanation"])


if __name__ == "__main__":
    main()
