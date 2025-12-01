# Supervisor

Agentic AI framework for powerful and flexible task completion.

## Overview

The Supervisor framework aims at enabling a simpler and more complex way of using LLMs as agents, particularly for complex tasks that require many iterations to complete.

Key features:

- **Self-modifying agents**: `Supervisor` agent can create tools and specialized agents at runtime
- **Configurable workflows**: Define multi-agent pipelines with specific configuration parameters
- **Log preprocessing**: Automatic SQLite ingestion and optional semantic search using embeddings
- **Evaluation framework**: Systematic testing of configurations on log analysis tasks
- **Flexible LLM support**: Works with Anthropic, OpenAI, and local models (Ollama)

### INTEND project

Supervisor is developed as a part of the INTEND project, which aims to transition from instruction-based to intent-based management of computing systems. In that context, the framework provides explanations of how systems interpreted and acted upon high-level intents, making AI-driven system management transparent and accountable.

**Current Focus**: Log analysis with agentic AI, studying different configurations and multi-agent workflows.

**Future Direction**: Integration with intent reports and automated system actions.

## Installation

```bash
# Clone repository
git clone https://github.com/ejhusom/supervisor
cd supervisor

# Install dependencies
pip install -r requirements.txt

# Optional: For semantic search
pip install sentence-transformers
```

## Quick start

### Basic usage

```bash
# Interactive mode
python main.py

# Single task
python main.py "Analyze error.log for patterns"
```

### With preprocessing

```bash
# Enable SQLite database for large logs
python main.py --preprocess workspace/data/logs.log

# With semantic search
python main.py --preprocess logs/*.log --embeddings
```

### Workflows

```bash
# Simple (default)
python main.py --workflow simple "Find errors"

# Evaluator (retry with feedback)
python main.py --workflow evaluator --max-iterations 3 "Analyze logs"

# Multi-stage (custom pipeline)
python main.py --workflow multi_stage \
  --stages "Parse logs" "Find patterns" "Write report" \
  "Analyze system logs"

# Predefined workflows
python main.py --workflow analysis "Investigate failures"
python main.py --workflow research "Study error trends"
```

## Configuration

Edit `config.toml`:

```toml
provider = "anthropic"  # or "openai", "ollama"
model = "claude-sonnet-4-20250514"
temperature = 0.0
max_tokens = 8192

workspace = "./workspace"
log_dir = "./logs"

# Preprocessing
embeddings_chunk_size = 100

# Tools (empty = all available)
tools_available = []
tools_unavailable = ["execute_python"]  # Disable specific tools
```

## Architecture

### Core components

**Supervisor**: Orchestrates tasks by creating tools and agents at runtime. Has meta-tools for self-modification and standard tools for execution.

**Agent**: LLM + tools + system prompt. Executes iterative tool-calling loops.

**Registries**: Dynamic tool and agent registries with persistence.

**Sandbox**: Safe Python and Unix command execution.

**Preprocessor**: Transforms large log files into queryable formats (SQLite, embeddings).

**Workflows**: Different execution strategies (simple, evaluator, multi-stage).

### Data flow

```
User Task → Workflow → Supervisor → [Create Tools/Agents] → Delegate → Results
                                  ↓
                             Preprocessing Tools
                             (SQLite, Embeddings)
```

## Workflows

### Simple workflow
Direct passthrough to supervisor. Default behavior.
```python
result = workflow.run("Analyze logs")
```

### Evaluator forkflow
Runs task, evaluates result, retries with feedback if needed.
```python
workflow = EvaluatorWorkflow(supervisor, max_iterations=3)
result = workflow.run("Find error patterns")
# Returns: result + evaluation_history + iterations_used
```

### Multi-Stage forkflow
Breaks task into sequential stages, accumulating context.
```python
workflow = MultiStageWorkflow(supervisor, stages=[
    "Parse log entries",
    "Identify error patterns",
    "Generate summary report"
])
result = workflow.run("Analyze system logs")
# Returns: result + stage_results
```

## Preprocessing

For large log files, enable preprocessing to create queryable databases:

```
# Command line
python main.py --preprocess logs/app.log --embeddings

# Programmatic
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG

preprocessor = Preprocessor()
preprocessor.add_step(SQLiteLogIngestion())
preprocessor.add_step(EmbeddingRAG(chunk_size=100))
preprocessor.process(Path("logs/app.log"))
```

Available tools after preprocessing:
- `query_logs_from_sqlite_database(sql)` - SQL queries
- `get_error_logs_from_sqlite_database(limit)` - Get errors
- `search_logs_from_sqlite_database(keyword)` - Keyword search
- `search_similar_from_embeddings(query)` - Semantic search

## Evaluation

Systematic testing of configurations on log analysis tasks:

```bash
# Generate ground truth
python eval_ground_truth.py logs/sample.log > ground_truth.json

# Run experiments
python eval_experiment.py \
  --log logs/sample.log \
  --ground-truth ground_truth.json \
  --config experiments.json \
  --output results.csv
```

See `docs/EVALUATION.md` for details.

## Examples

### Interactive session

```bash
$ python main.py
>>> Create a tool to count ERROR lines in logs
>>> Create an agent specialized in anomaly detection
>>> Analyze workspace/data/error.log for unusual patterns
```

### With preprocessing

```bash
$ python main.py --preprocess workspace/data/*.log --embeddings
>>> How many errors occurred?
>>> Find logs about connection timeouts
>>> What are the most common error patterns?
>>> Search for logs semantically similar to "database failure"
```

### Evaluator workflow

```bash
$ python main.py --workflow evaluator --max-iterations 3 \
    "Provide a detailed analysis of error trends in the logs"
# System will iterate until result is acceptable or max iterations reached
```

## API usage

```python
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry
from workflows import EvaluatorWorkflow

# Initialize
llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
supervisor = Supervisor(
    llm_client=llm,
    tool_registry=ToolRegistry(),
    agent_registry=AgentRegistry(),
    instructions_dir="instructions"
)

# Create workflow
workflow = EvaluatorWorkflow(supervisor, max_iterations=3)

# Execute task
result = workflow.run("Analyze logs for security issues")
print(result["content"])
```

## Project structure

```
supervisor/
├── core/                  # Core framework components
│   ├── agent.py          # Agent implementation
│   ├── supervisor.py     # Supervisor orchestration
│   ├── llm_client.py     # LLM interface
│   ├── sandbox.py        # Code execution
│   ├── preprocessor.py   # Log preprocessing
│   └── standard_tools.py # Built-in tools
├── workflows/            # Workflow implementations
│   ├── simple.py
│   ├── evaluator.py
│   └── multi_stage.py
├── registry/             # Dynamic registries
│   ├── tool_registry.py
│   └── agent_registry.py
├── eval/                 # Evaluation framework
│   ├── eval_ground_truth.py
│   ├── eval_experiment.py
│   └── eval_comparator.py
├── instructions/         # Agent guidance
├── workspace/            # Runtime workspace
│   ├── data/            # User data
│   ├── tools/           # Generated tools
│   └── agents/          # Generated agents
├── logs/                 # Execution logs
├── main.py              # Entry point
└── config.toml          # Configuration
```

## Testing

```bash
# Test components
python tests/test_workspace.py
python tests/test_workflows.py
python tests/test_eval_components.py

# Run all tests
pytest tests/
```

## Streamlit UI

```bash
streamlit run app.py
```

Features:
- Configure LLM settings
- Upload files
- Run queries
- View execution logs
- Interactive results

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Workflows](docs/workflows.md) - Workflow types and usage
- [Preprocessing](docs/preprocessing.md) - Log preprocessing system
- [Evaluation](docs/evaluation.md) - Evaluation framework
- [API Reference](docs/api.md) - Key APIs and interfaces

<!-- ## Contributing -->

<!-- This is a research project. For questions or collaboration: -->
<!-- - Review architecture documentation -->
<!-- - Check existing issues -->
<!-- - Test with your own log datasets -->

<!-- ## License -->

<!-- [Specify license] -->

<!-- ## Citation -->

<!-- If using Supervisor in research, please cite: -->

<!-- ``` -->
<!-- [To be added] -->
<!-- ``` -->

## Acknowledgments

Part of the INTEND project for intent-based system management.
