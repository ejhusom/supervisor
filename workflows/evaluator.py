"""
Evaluator workflow - runs supervisor, evaluates result, retries if needed.

Uses a separate evaluator agent to assess whether the supervisor's output
adequately addresses the task. If not acceptable, retries with feedback.
"""

import json
from typing import Dict, Any, Optional
from .base import Workflow
from core.agent import Agent


class EvaluatorWorkflow(Workflow):
    """
    Workflow with evaluation and retry logic.
    
    Flow:
    1. Run supervisor on task
    2. Evaluate result with evaluator agent
    3. If acceptable: return result
    4. If not: retry with feedback (up to max_iterations)
    
    The evaluator agent assesses whether the output adequately solves the task.
    """
    
    def __init__(self, supervisor, max_iterations: int = 3, verbose: bool = True):
        """
        Initialize evaluator workflow.
        
        Args:
            supervisor: Supervisor instance
            max_iterations: Maximum retry attempts (default: 3)
            verbose: Print evaluation feedback (default: True)
        """
        super().__init__(supervisor)
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Create evaluator agent
        self.evaluator = Agent(
            name="evaluator",
            system_prompt=self._get_evaluator_prompt(),
            llm_client=supervisor.llm_client,
            tools={},  # Evaluator doesn't need tools
            logging_enabled=False  # Don't log evaluator iterations
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute workflow with evaluation and retry.
        
        Args:
            task: User's task
            context: Optional context
            
        Returns:
            Dict with:
                - content: Final result
                - tool_calls: All tool calls across iterations
                - history: Full history
                - evaluation_history: List of evaluations
                - iterations_used: Number of supervisor runs
        """
        evaluation_history = []
        all_tool_calls = []
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Workflow Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*70}\n")
            
            # Run supervisor
            result = self.supervisor.run(task, context)
            all_tool_calls.extend(result.get("tool_calls", []))
            
            # Evaluate result
            eval_result = self._evaluate_result(task, result["content"])
            evaluation_history.append(eval_result)
            
            if self.verbose:
                self._print_evaluation(eval_result)
            
            # Check if acceptable
            if eval_result["acceptable"]:
                if self.verbose:
                    print("\n✓ Result accepted by evaluator\n")
                
                return {
                    "content": result["content"],
                    "tool_calls": all_tool_calls,
                    "history": result.get("history", []),
                    "evaluation_history": evaluation_history,
                    "iterations_used": iteration + 1,
                    "workflow": "evaluator"
                }
            
            # Not acceptable - prepare retry with feedback
            if self.verbose:
                print(f"\n✗ Result not accepted. Preparing retry...\n")
            
            # Enhance task with feedback for next iteration
            task = self._create_retry_task(
                original_task=task,
                previous_result=result["content"],
                feedback=eval_result
            )
        
        # Max iterations reached
        if self.verbose:
            print(f"\n- Max iterations ({self.max_iterations}) reached. Returning last result.\n")
        
        return {
            "content": result["content"],
            "tool_calls": all_tool_calls,
            "history": result.get("history", []),
            "evaluation_history": evaluation_history,
            "iterations_used": self.max_iterations,
            "workflow": "evaluator",
            "max_iterations_reached": True
        }
    
    def _evaluate_result(self, task: str, result: str) -> Dict[str, Any]:
        """
        Evaluate whether result adequately addresses task.
        
        Args:
            task: Original task
            result: Supervisor's result
            
        Returns:
            Dict with acceptable (bool), reasoning, and suggestions
        """
        eval_prompt = f"""Task: {task}

Result produced:
{result}

Evaluate whether this result adequately addresses the task. Respond ONLY with valid JSON in this exact format:
{{
    "acceptable": true or false,
    "reasoning": "brief explanation of why acceptable or not",
    "suggestions": "specific improvements needed (if not acceptable) or empty string"
}}"""
        
        eval_agent_result = self.evaluator.run(eval_prompt, max_iterations=3)
        
        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = eval_agent_result["content"]
            
            # Try to find JSON in the response
            if "```json" in response_text:
                # Extract from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            else:
                # Try to parse entire response
                json_text = response_text.strip()
            
            evaluation = json.loads(json_text)
            
            # Validate required fields
            if "acceptable" not in evaluation:
                raise ValueError("Missing 'acceptable' field")
            
            return {
                "acceptable": bool(evaluation["acceptable"]),
                "reasoning": evaluation.get("reasoning", "No reasoning provided"),
                "suggestions": evaluation.get("suggestions", "")
            }
        
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: simple heuristic based on response text
            response_lower = eval_agent_result["content"].lower()
            
            # Look for acceptance indicators
            acceptable = any(indicator in response_lower for indicator in [
                "acceptable", "adequate", "satisfactory", "good", "correct", "yes"
            ])
            
            return {
                "acceptable": acceptable,
                "reasoning": f"Could not parse JSON evaluation (error: {e}). Used heuristic.",
                "suggestions": eval_agent_result["content"] if not acceptable else ""
            }
    
    def _create_retry_task(
        self,
        original_task: str,
        previous_result: str,
        feedback: Dict[str, Any]
    ) -> str:
        """
        Create enhanced task for retry with feedback.
        
        Args:
            original_task: Original user task
            previous_result: Previous supervisor result
            feedback: Evaluation feedback
            
        Returns:
            Enhanced task description
        """
        retry_task = f"""{original_task}

PREVIOUS ATTEMPT FEEDBACK:
The previous attempt was not acceptable. Here's what needs improvement:

Reasoning: {feedback['reasoning']}
Suggestions: {feedback['suggestions']}

Previous result:
{previous_result[:500]}{'...' if len(previous_result) > 500 else ''}

Please address the feedback and produce an improved result."""
        
        return retry_task
    
    def _print_evaluation(self, eval_result: Dict[str, Any]) -> None:
        """Print evaluation feedback."""
        print(f"Evaluation:")
        print(f"  Acceptable: {eval_result['acceptable']}")
        print(f"  Reasoning: {eval_result['reasoning']}")
        if eval_result['suggestions']:
            print(f"  Suggestions: {eval_result['suggestions']}")
    
    def _get_evaluator_prompt(self) -> str:
        """Get system prompt for evaluator agent."""
        return """You are an evaluator agent that assesses whether a result adequately addresses a given task.

Your role:
1. Read the task description carefully
2. Examine the result produced
3. Determine if the result adequately solves the task
4. Provide specific, actionable feedback

Evaluation criteria:
- Completeness: Does it address all parts of the task?
- Correctness: Is the information/analysis accurate?
- Clarity: Is the result well-structured and understandable?
- Relevance: Does it stay focused on the task?

You MUST respond with valid JSON in this exact format:
{
    "acceptable": true or false,
    "reasoning": "brief explanation",
    "suggestions": "specific improvements (if not acceptable)"
}

Be critical but fair. Set a high bar for quality, but recognize when work is good enough.
If acceptable, set suggestions to empty string. If not acceptable, provide specific, actionable suggestions."""
