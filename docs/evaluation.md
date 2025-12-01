# Evaluation

## Overview

Systematic framework for testing iExplain configurations on log analysis tasks.

## Quick Start

```bash
# 1. Generate ground truth
python eval/eval_ground_truth.py logs/sample.log > ground_truth.json

# 2. Run experiments
python eval/eval_experiment.py \
  --log logs/sample.log \
  --ground-truth ground_truth.json \
  --output results.csv
```

## Components

### Ground Truth Generator

Extracts verifiable statistics from logs and generates evaluation questions.

**Usage**:
```bash
python eval/eval_ground_truth.py logs/app.log > gt.json
```

**Output**:
```json
{
  "metadata": {...},
  "statistics": {
    "total_lines": 1000,
    "level_counts": {"ERROR": 50, "INFO": 900},
    "component_counts": {...},
    ...
  },
  "evaluation_questions": [
    {
      "id": "q1_error_count",
      "question": "How many ERROR level entries?",
      "answer_type": "integer",
      "expected": 50,
      "tolerance": 0
    },
    ...
  ]
}
```

**Question Types**:
- `integer` - Numeric with tolerance
- `string_match` - Text matching
- `list` - Collection of items

### Experiment Runner

Tests configurations on ground truth questions.

**Usage**:
```bash
python eval/eval_experiment.py \
  --log logs/app.log \
  --ground-truth gt.json \
  --config experiments.json \
  --output results.csv
```

**Default configs**:
- `baseline_simple` - No preprocessing
- `preprocessing_simple` - With SQLite
- `evaluator` - Retry workflow
- `multi_stage` - Pipeline workflow

### Comparison Logic

Flexible answer comparison:

**Integer**: Exact or within tolerance
```python
answer=148, expected=150, tolerance=2 → ✓
answer=145, expected=150, tolerance=2 → ✗
```

**String**: Case-insensitive substring
```python
answer="KERNEL", expected=["kernel"] → ✓
answer="The component is KERNEL", expected=["kernel"] → ✓
```

**List**: Set-based or ordered
```python
answer=["A", "B", "C"], expected=["C", "B", "A"], order_matters=False → ✓
answer=["A", "B", "C"], expected=["C", "B", "A"], order_matters=True → ✗
```

## Configuration

Create `experiments.json`:

```json
[
  {
    "name": "gpt4o_baseline",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "workflow": "simple",
    "use_preprocessing": false
  },
  {
    "name": "claude_with_preprocessing",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "workflow": "simple",
    "use_preprocessing": true,
    "use_embeddings": true
  },
  {
    "name": "evaluator_workflow",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "workflow": "evaluator",
    "use_preprocessing": true,
    "max_iterations": 3
  }
]
```

**Parameters**:
- `name` - Unique identifier
- `provider` - LLM provider
- `model` - Model name
- `workflow` - Workflow type
- `use_preprocessing` - Enable SQLite
- `use_embeddings` - Enable semantic search
- `max_iterations` - For evaluator workflow

## Results Format

CSV with columns:
- `config_name` - Configuration identifier
- `provider`, `model`, `workflow` - Config details
- `preprocessing`, `embeddings` - Features
- `question_id`, `question` - Question details
- `expected`, `answer` - Expected vs actual
- `correct` - Boolean result
- `reason` - Comparison explanation
- `time_seconds` - Duration
- `error` - Error message if failed

## Analysis Examples

### Calculate Accuracy

```python
import pandas as pd

df = pd.read_csv('results.csv')

# Overall accuracy
accuracy = df.groupby('config_name')['correct'].mean() * 100
print(accuracy)

# By workflow
workflow_acc = df.groupby('workflow')['correct'].mean() * 100
print(workflow_acc)
```

### Performance Comparison

```python
# Accuracy vs time trade-off
summary = df.groupby('config_name').agg({
    'correct': 'mean',
    'time_seconds': 'mean',
    'preprocessing': 'first'
})
print(summary)
```

### Identify Difficult Questions

```python
# Questions with low success rate
question_diff = df.groupby('question_id').agg({
    'correct': 'mean',
    'question': 'first'
})
hard = question_diff[question_diff['correct'] < 0.5]
print(hard)
```

## Example Workflow

```bash
# 1. Prepare data
cp my_logs.log workspace/data/

# 2. Generate ground truth
python eval/eval_ground_truth.py workspace/data/my_logs.log > gt.json

# 3. Create experiment config (or use default)
cat > my_experiments.json << EOF
[
  {
    "name": "baseline",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "workflow": "simple",
    "use_preprocessing": false
  },
  {
    "name": "with_preprocessing",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "workflow": "simple",
    "use_preprocessing": true
  }
]
EOF

# 4. Run experiments
python eval/eval_experiment.py \
  --log workspace/data/my_logs.log \
  --ground-truth gt.json \
  --config my_experiments.json \
  --output my_results.csv

# 5. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('my_results.csv')
print(df.groupby('config_name')['correct'].mean() * 100)
"
```

## Custom Ground Truth

Edit `eval/eval_ground_truth.py` to add questions:

```python
def generate_evaluation_questions(stats):
    questions = [
        {
            "id": "custom_q1",
            "question": "What is the error rate?",
            "answer_type": "integer",
            "expected": compute_error_rate(stats),
            "tolerance": 1
        },
        ...
    ]
    return questions
```

## Testing Components

```bash
# Test JSON extraction and comparison
python tests/test_eval_components.py
```

## Performance Notes

**Per question**: 5-30 seconds depending on:
- LLM speed
- Workflow complexity
- Preprocessing overhead (first query only)

**Full experiment**: 10-20 minutes for:
- 4 configurations
- 8 questions each
- = 32 total evaluations

**Use `--quiet` to reduce output overhead**

## Best Practices

1. **Start small**: Test 2-3 configs before full suite
2. **Incremental results**: Results saved after each question
3. **Monitor costs**: Track API usage during experiments
4. **Check errors**: Review `error` column for failures
5. **Iterate**: Refine questions based on results

## Extending

### Add Question Type

In `eval/eval_comparator.py`:

```python
def _compare_new_type(self, answer, expected, spec):
    # Your comparison logic
    return {
        "correct": bool,
        "reason": str,
        "answer": answer,
        "expected": expected
    }
```

### Support New Log Format

In `eval/eval_ground_truth.py`:

```python
class CustomLogParser:
    @staticmethod
    def parse_line(line):
        # Your parsing logic
        return {"timestamp": ..., "level": ..., "message": ...}
```

## Troubleshooting

**"Could not extract JSON"**:
- LLM didn't return valid JSON
- Check structured query prompt
- Review actual LLM response

**High error rate**:
- Verify API keys in config.toml
- Check log format matches parser
- Review question difficulty

**Slow execution**:
- Use faster model (gpt-4o-mini)
- Reduce max_iterations for evaluator
- Disable embeddings if not needed
