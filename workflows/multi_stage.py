"""
Multi-stage workflow - breaks tasks into sequential stages.

Useful for complex tasks that benefit from being broken down into steps,
where each stage's output feeds into the next stage as context.
"""

from typing import Dict, Any, Optional, List
from .base import Workflow


class MultiStageWorkflow(Workflow):
    """
    Workflow that executes a task in sequential stages.
    
    Each stage:
    1. Receives the original task + stage description
    2. Has access to all previous stage results as context
    3. Produces output that accumulates in context for next stages
    
    Useful for:
    - Complex analysis pipelines (parse → analyze → summarize)
    - Multi-step transformations (extract → process → format)
    - Research workflows (gather → synthesize → conclude)
    """
    
    def __init__(
        self,
        supervisor,
        stages: List[str],
        accumulate_context: bool = True,
        verbose: bool = True
    ):
        """
        Initialize multi-stage workflow.
        
        Args:
            supervisor: Supervisor instance
            stages: List of stage descriptions (e.g., ["Parse the logs", "Analyze errors", "Write summary"])
            accumulate_context: Pass previous results to later stages (default: True)
            verbose: Print stage progress (default: True)
        """
        super().__init__(supervisor)
        self.stages = stages
        self.accumulate_context = accumulate_context
        self.verbose = verbose
        
        if not stages:
            raise ValueError("Must provide at least one stage")
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute workflow through all stages.
        
        Args:
            task: User's original task
            context: Optional initial context
            
        Returns:
            Dict with:
                - content: Final stage's result
                - tool_calls: All tool calls across all stages
                - history: Combined history from all stages
                - stage_results: List of results from each stage
                - workflow: "multi_stage"
        """
        stage_results = []
        all_tool_calls = []
        accumulated_context = context.copy() if context else {}
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Multi-Stage Workflow: {len(self.stages)} stages")
            print(f"{'='*70}\n")
        
        for i, stage_desc in enumerate(self.stages, 1):
            if self.verbose:
                print(f"\n{'─'*70}")
                print(f"Stage {i}/{len(self.stages)}: {stage_desc}")
                print(f"{'─'*70}\n")
            
            # Construct stage task
            stage_task = self._create_stage_task(
                original_task=task,
                stage_desc=stage_desc,
                stage_num=i,
                total_stages=len(self.stages),
                accumulated_context=accumulated_context if self.accumulate_context else {}
            )
            
            # Execute stage
            result = self.supervisor.run(stage_task, accumulated_context)
            
            # Collect results
            stage_results.append({
                "stage_num": i,
                "stage_desc": stage_desc,
                "content": result["content"],
                "tool_calls": result.get("tool_calls", [])
            })
            
            all_tool_calls.extend(result.get("tool_calls", []))
            
            # Accumulate context for next stage
            if self.accumulate_context:
                accumulated_context[f"stage_{i}_result"] = result["content"]
                accumulated_context[f"stage_{i}_description"] = stage_desc
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"All stages complete")
            print(f"{'='*70}\n")
        
        # Return final stage result as main output
        final_result = stage_results[-1]
        
        return {
            "content": final_result["content"],
            "tool_calls": all_tool_calls,
            "history": [],  # Could combine all histories if needed
            "stage_results": stage_results,
            "workflow": "multi_stage",
            "stages_completed": len(self.stages)
        }
    
    def _create_stage_task(
        self,
        original_task: str,
        stage_desc: str,
        stage_num: int,
        total_stages: int,
        accumulated_context: Dict[str, Any]
    ) -> str:
        """
        Create task description for a specific stage.
        
        Args:
            original_task: User's original task
            stage_desc: Description of this stage
            stage_num: Current stage number
            total_stages: Total number of stages
            accumulated_context: Results from previous stages
            
        Returns:
            Task description for this stage
        """
        task_parts = [
            f"ORIGINAL TASK: {original_task}",
            f"",
            f"CURRENT STAGE ({stage_num}/{total_stages}): {stage_desc}",
        ]
        
        # Add context from previous stages if available
        if accumulated_context and stage_num > 1:
            task_parts.append("")
            task_parts.append("PREVIOUS STAGE RESULTS:")
            
            for key, value in accumulated_context.items():
                if key.startswith("stage_") and key.endswith("_result"):
                    stage_number = key.split("_")[1]
                    desc_key = f"stage_{stage_number}_description"
                    desc = accumulated_context.get(desc_key, "Unknown stage")
                    
                    # Truncate long results
                    value_str = str(value)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "... [truncated]"
                    
                    task_parts.append(f"  Stage {stage_number} ({desc}):")
                    task_parts.append(f"    {value_str}")
        
        task_parts.append("")
        task_parts.append(f"Focus on completing stage {stage_num}: {stage_desc}")
        
        return "\n".join(task_parts)


class PredefinedMultiStageWorkflow(MultiStageWorkflow):
    """
    Convenience class for common multi-stage workflows.
    
    Provides predefined stage templates for common patterns.
    """
    
    @classmethod
    def analysis_workflow(cls, supervisor) -> 'PredefinedMultiStageWorkflow':
        """
        Create a workflow for: parse → analyze → summarize.
        
        Good for log analysis, data investigation, etc.
        """
        stages = [
            "Parse and extract relevant data from the input",
            "Analyze the extracted data to identify patterns, anomalies, and insights",
            "Summarize findings in a clear, structured report"
        ]
        return cls(supervisor, stages=stages)
    
    @classmethod
    def research_workflow(cls, supervisor) -> 'PredefinedMultiStageWorkflow':
        """
        Create a workflow for: gather → synthesize → conclude.
        
        Good for research tasks, information synthesis, etc.
        """
        stages = [
            "Gather relevant information and data related to the task",
            "Synthesize the gathered information into coherent insights",
            "Draw conclusions and provide recommendations"
        ]
        return cls(supervisor, stages=stages)
    
    @classmethod
    def transformation_workflow(cls, supervisor) -> 'PredefinedMultiStageWorkflow':
        """
        Create a workflow for: extract → transform → format.
        
        Good for data processing, format conversion, etc.
        """
        stages = [
            "Extract the necessary data from the input",
            "Transform and process the data as needed",
            "Format the output in the desired structure"
        ]
        return cls(supervisor, stages=stages)
