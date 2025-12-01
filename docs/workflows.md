# Workflows

## Overview

Workflows define **how** the supervisor executes tasks, while the supervisor defines **what** gets done.

## Workflow Types

### Simple Workflow

Default behavior - direct passthrough to supervisor.

**Use when**: Task doesn't need evaluation or staging.

```bash
python main.py --workflow simple "Analyze logs"
```

```python
from workflows import SimpleWorkflow

workflow = SimpleWorkflow(supervisor)
result = workflow.run("Find errors")
# Returns: {content, tool_calls, history}
```

### Evaluator Workflow

Runs task, evaluates result quality, retries with feedback.

**Use when**: Need high-quality results, willing to use more iterations.

```bash
python main.py --workflow evaluator --max-iterations 3 "Detailed error analysis"
```

```python
from workflows import EvaluatorWorkflow

workflow = EvaluatorWorkflow(supervisor, max_iterations=3, verbose=True)
result = workflow.run("Analyze security logs")
# Returns: {content, evaluation_history, iterations_used, ...}
```

**Flow**:
1. Supervisor runs task
2. Evaluator agent assesses result (JSON: acceptable, reasoning, suggestions)
3. If acceptable: return
4. If not: retry with feedback
5. Repeat up to max_iterations

**Evaluation criteria**:
- Completeness
- Correctness
- Clarity
- Relevance

### Multi-Stage Workflow

Breaks task into sequential stages, accumulating context.

**Use when**: Task benefits from decomposition (parse → analyze → report).

```bash
python main.py --workflow multi_stage \
  --stages "Parse log entries" "Identify patterns" "Write summary" \
  "Analyze system logs"
```

```python
from workflows import MultiStageWorkflow

workflow = MultiStageWorkflow(supervisor, stages=[
    "Extract relevant log data",
    "Perform statistical analysis",
    "Generate recommendations"
], accumulate_context=True)

result = workflow.run("Investigate failures")
# Returns: {content, stage_results, stages_completed, ...}
```

**Context accumulation**:
Each stage receives previous results as context:
```python
{
    "stage_1_result": "...",
    "stage_1_description": "...",
    "stage_2_result": "...",
    ...
}
```

### Predefined Workflows

Convenience workflows for common patterns:

**Analysis**: parse → analyze → summarize
```python
from workflows import PredefinedMultiStageWorkflow

workflow = PredefinedMultiStageWorkflow.analysis_workflow(supervisor)
```
```bash
python main.py --workflow analysis "Study error trends"
```

**Research**: gather → synthesize → conclude
```bash
python main.py --workflow research "Compare error rates"
```

**Transformation**: extract → transform → format
```bash
python main.py --workflow transformation "Convert logs to report"
```

## Choosing a Workflow

| Workflow | Use Case |
|----------|----------|
| Simple | Quick queries, straightforward tasks |
| Evaluator | High quality needs, complex analysis |
| Multi-Stage | Decomposable tasks, pipelines |

## Custom Workflows

Extend `Workflow` base class:

```python
from workflows.base import Workflow

class CustomWorkflow(Workflow):
    def run(self, task, context=None):
        # Your orchestration logic
        result = self.supervisor.run(task, context)
        return result
```

## Workflow Results

All workflows return dict with:
- `content` - Final result text
- `tool_calls` - All tool calls made
- `history` - Agent interaction history
- Workflow-specific fields

**SimpleWorkflow**:
```python
{
    "content": str,
    "tool_calls": list,
    "history": list
}
```

**EvaluatorWorkflow**:
```python
{
    "content": str,
    "tool_calls": list,
    "history": list,
    "evaluation_history": [
        {"acceptable": bool, "reasoning": str, "suggestions": str}
    ],
    "iterations_used": int,
    "max_iterations_reached": bool  # if applicable
}
```

**MultiStageWorkflow**:
```python
{
    "content": str,  # Final stage result
    "tool_calls": list,  # All stages combined
    "history": list,
    "stage_results": [
        {"stage_num": int, "stage_desc": str, "content": str, "tool_calls": list}
    ],
    "stages_completed": int
}
```

## Performance Considerations

**Simple**: Fastest, single supervisor call
**Evaluator**: 1-N supervisor calls + N evaluator calls
**Multi-Stage**: N supervisor calls (one per stage)

Cost scales with:
- Number of iterations/stages
- Max tokens per call
- Model pricing

## Examples

### Evaluator with Retry

```python
workflow = EvaluatorWorkflow(supervisor, max_iterations=3)
result = workflow.run("Provide comprehensive log analysis")

print(f"Used {result['iterations_used']} iterations")
for i, eval in enumerate(result['evaluation_history'], 1):
    print(f"Iteration {i}: {'✓' if eval['acceptable'] else '✗'}")
```

### Multi-Stage Pipeline

```python
workflow = MultiStageWorkflow(supervisor, stages=[
    "Parse OpenStack logs",
    "Extract error patterns",
    "Correlate with system events",
    "Generate incident report"
])

result = workflow.run("Investigate yesterday's outage")

for stage in result['stage_results']:
    print(f"Stage {stage['stage_num']}: {stage['stage_desc']}")
    print(f"  Tools: {len(stage['tool_calls'])}")
```

### Combining Approaches

```python
# Multi-stage where each stage uses evaluator
for stage_desc in stages:
    stage_workflow = EvaluatorWorkflow(supervisor, max_iterations=2)
    stage_result = stage_workflow.run(stage_desc)
    # Accumulate results
```

## Best Practices

1. **Start simple**: Use SimpleWorkflow first, upgrade if needed
2. **Limit iterations**: Set reasonable max_iterations (3-5)
3. **Clear stages**: Make stage descriptions specific and actionable
4. **Monitor costs**: More iterations = more API calls
5. **Log everything**: Enable verbose mode during development
