#!/usr/bin/env python3
"""
Experiment runner for evaluating iExplain on log analysis tasks.

Usage:
    python eval_experiment.py --log <log_file> --ground-truth <gt.json> --output results.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry
from workflows import SimpleWorkflow, EvaluatorWorkflow, MultiStageWorkflow

from eval_structured_query import query_question
from eval_comparator import evaluate_answer


class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    def __init__(
        self,
        name: str,
        provider: str,
        model: str,
        workflow: str = "simple",
        use_preprocessing: bool = False,
        use_embeddings: bool = False,
        **kwargs
    ):
        self.name = name
        self.provider = provider
        self.model = model
        self.workflow = workflow
        self.use_preprocessing = use_preprocessing
        self.use_embeddings = use_embeddings
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "workflow": self.workflow,
            "use_preprocessing": self.use_preprocessing,
            "use_embeddings": self.use_embeddings,
            **self.extra_params
        }


class ExperimentRunner:
    """Runs evaluation experiments with different configurations."""
    
    def __init__(self, log_file: Path, ground_truth: Dict, verbose: bool = True):
        self.log_file = log_file
        self.ground_truth = ground_truth
        self.verbose = verbose
        self.results = []
    
    def run_experiment(self, exp_config: ExperimentConfig) -> List[Dict]:
        """Run experiment with a specific configuration."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Running: {exp_config.name}")
            print(f"{'='*70}")
            print(f"Provider: {exp_config.provider}")
            print(f"Model: {exp_config.model}")
            print(f"Workflow: {exp_config.workflow}")
            print(f"Preprocessing: {exp_config.use_preprocessing}")
            print(f"Embeddings: {exp_config.use_embeddings}")
            print()

        # Clearing workspace before experiment. Delete previously created agents from workspace/agents and tools from workspace/tools. Only keep the log file and related databases in workspace/data. Get folder names from config.py.
        workspace = config.get("workspace")
        workspace_agents = config.get("workspace_agents")
        workspace_tools = config.get("workspace_tools")
        workspace_data = config.get("workspace_data")
        for folder in [workspace_agents, workspace_tools]:
            for item in Path(folder).iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        if subitem.is_file():
                            subitem.unlink()
                            print(f"Deleted file: {subitem}")
                    item.rmdir()
                    print(f"Deleted folder: {item}")
                elif item.is_file():
                    item.unlink()
                    print(f"Deleted file: {item}")

        # Clear data directory, but keep log files and related DBs
        for item in Path(workspace_data).iterdir():
            if item.is_file() and not item.name.startswith(self.log_file.stem):
                item.unlink()
                print(f"Deleted file: {item}")
            elif item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_file() and not subitem.name.startswith(self.log_file.stem):
                        subitem.unlink()
                        print(f"Deleted file: {subitem}")
                if not any(item.iterdir()):
                    item.rmdir()
                    print(f"Deleted folder: {item}")

        # Setup preprocessing if needed
        preprocessor = None
        if exp_config.use_preprocessing:
            if self.verbose:
                print("Setting up preprocessing...")
            preprocessor = self._setup_preprocessing(exp_config.use_embeddings)
            if self.verbose:
                print("Preprocessing complete\n")
        
        # Initialize supervisor
        if self.verbose:
            print("Initializing supervisor...")
        
        supervisor = self._create_supervisor(
            provider=exp_config.provider,
            model=exp_config.model,
            preprocessor=preprocessor
        )
        
        # Create workflow
        workflow = self._create_workflow(exp_config.workflow, supervisor)
        
        if self.verbose:
            print(f"Ready. Querying {len(self.ground_truth['evaluation_questions'])} questions...\n")
        
        # Run all questions
        experiment_results = []
        questions = self.ground_truth['evaluation_questions']
        
        for i, question in enumerate(questions, 1):
            # Add name of file to question context
            question['question'] = f"Solve the following task based on this file: {self.log_file.name}.\n\n{question['question']}"

            if self.verbose:
                print(f"[{i}/{len(questions)}] {question['id']}: {question['question'][:60]}...")
            
            # Query and evaluate
            result = self._query_and_evaluate(workflow, question, exp_config)
            experiment_results.append(result)
            
            if self.verbose:
                status = "✓" if result['correct'] else "✗"
                print(f"    {status} {result['reason']}")
                if result['tokens_total'] > 0:
                    print(f"    Tokens: {result['tokens_total']} ({result['tokens_input']} in, {result['tokens_output']} out)")
        
        if self.verbose:
            # Summary
            correct = sum(1 for r in experiment_results if r['correct'])
            total = len(experiment_results)
            accuracy = (correct / total * 100) if total > 0 else 0
            total_tokens = sum(r['tokens_total'] for r in experiment_results)
            print(f"\n  Results: {correct}/{total} correct ({accuracy:.1f}%)")
            print(f"  Total tokens: {total_tokens:,}")
        
        return experiment_results
    
    def _setup_preprocessing(self, use_embeddings: bool) -> Preprocessor:
        """Setup preprocessing on log file."""
        preprocessor = Preprocessor()
        preprocessor.add_step(SQLiteLogIngestion())
        
        if use_embeddings:
            try:
                chunk_size = config.get("embeddings_chunk_size", 100)
                preprocessor.add_step(EmbeddingRAG(chunk_size=chunk_size))
            except ImportError:
                if self.verbose:
                    print("  Warning: Embeddings not available")
        
        preprocessor.process(self.log_file)
        return preprocessor
    
    def _create_supervisor(self, provider: str, model: str, preprocessor: Preprocessor = None) -> Supervisor:
        """Create supervisor with given configuration."""
        # Try to get API key from environment
        if provider in ["anthropic", "openai"]:
            api_key = os.environ.get("ANTHROPIC_API_KEY") if provider == "anthropic" else os.environ.get("OPENAI_API_KEY")
        else:
            api_key = None

        llm_client = LLMClient(provider=provider, model=model, api_key=api_key)
        tool_registry = ToolRegistry()
        agent_registry = AgentRegistry()
        
        return Supervisor(
            llm_client=llm_client,
            tool_registry=tool_registry,
            agent_registry=agent_registry,
            instructions_dir="instructions",
            preprocessor=preprocessor
        )
    
    def _create_workflow(self, workflow_type: str, supervisor):
        """Create workflow of specified type."""
        if workflow_type == "evaluator":
            return EvaluatorWorkflow(supervisor, max_iterations=3, verbose=False)
        elif workflow_type == "multi_stage":
            stages = [
                "Parse and extract relevant data from the log file",
                "Analyze the extracted data to answer the question",
                "Provide a clear, structured answer"
            ]
            return MultiStageWorkflow(supervisor, stages=stages, verbose=False)
        else:
            return SimpleWorkflow(supervisor)
    
    def _extract_token_usage(self, supervisor) -> tuple:
        """
        Extract token usage from supervisor's logger.
        
        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens)
        """
        try:
            from core.logger import get_logger
            logger = get_logger()
            
            if not logger or not logger.current_log:
                return 0, 0, 0
            
            # Sum tokens across all interactions and iterations
            total_input = 0
            total_output = 0
            
            for interaction in logger.current_log.get('interactions', []):
                for iteration in interaction.get('iterations', []):
                    usage = iteration.get('model_info', {}).get('usage', {})
                    total_input += usage.get('input_tokens', 0)
                    total_output += usage.get('output_tokens', 0)
            
            return total_input, total_output, total_input + total_output
            
        except Exception:
            # Fallback if logger access fails
            return 0, 0, 0
    
    def _query_and_evaluate(self, workflow, question: Dict, exp_config: ExperimentConfig) -> Dict:
        """Query a single question and evaluate the answer."""
        start_time = time.time()
        # Format start_time for logging in ISO format
        start_time_iso = datetime.fromtimestamp(start_time).isoformat()

        
        try:
            answer = query_question(workflow.supervisor, question)
            eval_result = evaluate_answer(answer, question)
            elapsed_time = time.time() - start_time
            
            # Extract token usage
            tokens_in, tokens_out, tokens_total = self._extract_token_usage(workflow.supervisor)
            
            return {
                "start_time": start_time_iso,
                "config_name": exp_config.name,
                "provider": exp_config.provider,
                "model": exp_config.model,
                "workflow": exp_config.workflow,
                "preprocessing": exp_config.use_preprocessing,
                "embeddings": exp_config.use_embeddings,
                "question_id": question['id'],
                "question": question['question'],
                "answer_type": question['answer_type'],
                "expected": question['expected'],
                "answer": eval_result['answer'],
                "correct": eval_result['correct'],
                "reason": eval_result['reason'],
                "time_seconds": elapsed_time,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "tokens_total": tokens_total,
                "error": None
            }
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "start_time": start_time_iso,
                "config_name": exp_config.name,
                "provider": exp_config.provider,
                "model": exp_config.model,
                "workflow": exp_config.workflow,
                "preprocessing": exp_config.use_preprocessing,
                "embeddings": exp_config.use_embeddings,
                "question_id": question['id'],
                "question": question['question'],
                "answer_type": question['answer_type'],
                "expected": question['expected'],
                "answer": None,
                "correct": False,
                "reason": f"Error: {str(e)}",
                "time_seconds": elapsed_time,
                "tokens_input": 0,
                "tokens_output": 0,
                "tokens_total": 0,
                "error": str(e)
            }
    
    def run_all_experiments(self, configs: List[ExperimentConfig]) -> List[Dict]:
        """Run all experiment configurations."""
        all_results = []
        
        for i, config in enumerate(configs, 1):
            if self.verbose:
                print(f"\n{'#'*70}")
                print(f"# Experiment {i}/{len(configs)}")
                print(f"{'#'*70}")
            
            results = self.run_experiment(config)
            all_results.extend(results)
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_file: Path):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save")
            return
        
        fieldnames = [
            "start_time", "config_name", "provider", "model", "workflow",
            "preprocessing", "embeddings",
            "question_id", "question", "answer_type",
            "expected", "answer", "correct", "reason",
            "time_seconds", "tokens_input", "tokens_output", "tokens_total",
            "error"
        ]
        
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        except Exception as e:
            # Try alternative saving method
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_file = output_file.parent / f"results_{timestamp}.csv"
            with open(alt_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        
        if self.verbose:
            print(f"\nResults saved to: {output_file}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            return
        
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}\n")
        
        # Group by config
        configs = {}
        for result in self.results:
            config_name = result['config_name']
            if config_name not in configs:
                configs[config_name] = []
            configs[config_name].append(result)
        
        # Print per-config summary
        for config_name, results in configs.items():
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            accuracy = (correct / total * 100) if total > 0 else 0
            avg_time = sum(r['time_seconds'] for r in results) / total if total > 0 else 0
            total_tokens = sum(r['tokens_total'] for r in results)
            avg_tokens = total_tokens / total if total > 0 else 0
            
            print(f"{config_name}:")
            print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
            print(f"  Avg time: {avg_time:.2f}s per question")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Avg tokens: {avg_tokens:.0f} per question")
            print()


def load_default_configs() -> List[ExperimentConfig]:
    """Load default experiment configurations."""
    provider = config.get("provider", "openai")
    model = config.get("model", "gpt-4o-mini")
    
    return [
        ExperimentConfig(
            name="baseline_simple",
            provider=provider,
            model=model,
            workflow="simple",
            use_preprocessing=False
        ),
        ExperimentConfig(
            name="preprocessing_simple",
            provider=provider,
            model=model,
            workflow="simple",
            use_preprocessing=True
        ),
        ExperimentConfig(
            name="evaluator",
            provider=provider,
            model=model,
            workflow="evaluator",
            use_preprocessing=True
        ),
        ExperimentConfig(
            name="multi_stage",
            provider=provider,
            model=model,
            workflow="multi_stage",
            use_preprocessing=True
        ),
    ]


def load_configs_from_file(config_file: Path) -> List[ExperimentConfig]:
    """Load experiment configurations from JSON file."""
    with open(config_file) as f:
        configs_data = json.load(f)
    
    configs = []
    for cfg_data in configs_data:
        configs.append(ExperimentConfig(**cfg_data))
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation experiments on iExplain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configurations
  python eval_experiment.py --log logs/bgl_2k.log --ground-truth gt.json
  
  # Specify output file
  python eval_experiment.py --log logs/bgl_2k.log --ground-truth gt.json --output results.csv
  
  # Use custom config file
  python eval_experiment.py --log logs/bgl_2k.log --ground-truth gt.json --config experiments.json
  
  # Quiet mode
  python eval_experiment.py --log logs/bgl_2k.log --ground-truth gt.json --quiet
        """
    )
    
    parser.add_argument('--log', '-l', required=True, help='Log file to analyze')
    parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth JSON file')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--config', '-c', help='Custom experiment config JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    # Validate inputs
    log_file = Path(args.log)
    gt_file = Path(args.ground_truth)
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    if not gt_file.exists():
        print(f"Error: Ground truth file not found: {gt_file}")
        sys.exit(1)
    
    # Load ground truth
    with open(gt_file) as f:
        ground_truth = json.load(f)
    
    # Create experiment runner
    runner = ExperimentRunner(
        log_file=log_file,
        ground_truth=ground_truth,
        verbose=not args.quiet
    )
    
    # Load configurations
    if args.config:
        configs = load_configs_from_file(Path(args.config))
    else:
        configs = load_default_configs()
    
    if not args.quiet:
        print("="*70)
        print("iExplain Evaluation Experiment")
        print("="*70)
        print(f"Log file: {log_file}")
        print(f"Ground truth: {gt_file}")
        print(f"Questions: {len(ground_truth['evaluation_questions'])}")
        print(f"Configurations: {len(configs)}")
        print()
        
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.name}: {cfg.workflow} workflow, "
                  f"preprocessing={cfg.use_preprocessing}")
    
    # Run experiments
    start_time = time.time()
    runner.run_all_experiments(configs)
    total_time = time.time() - start_time
    
    # Save results
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{timestamp}.csv"

    output_file = Path(args.output)
    runner.save_results(output_file)
    
    # Print summary
    if not args.quiet:
        runner.print_summary()
        print(f"Total experiment time: {total_time:.1f}s")
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
