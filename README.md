# Supervisor

A self-modifying agentic AI framework where a Supervisor agent orchestrates complex tasks by dynamically creating tools and specialized agents at runtime.

## Overview

**Core Concept**: The Supervisor extends its capabilities through meta-tools (`create_tool`, `create_agent`, `delegate_to_agent`), distinguishing it from standard task execution frameworks.

**Key Features**:
- Dynamic tool and agent creation at runtime
- Configurable workflows (simple, evaluator, multi-stage)
- Log preprocessing with SQLite and semantic search
- Evaluation framework for systematic testing
- Flexible LLM support (Anthropic, OpenAI, Ollama)

**INTEND Project Context**: Developed for intent-based management of computing systems. Current focus is log analysis with agentic AI. Future direction includes integration with intent reports and automated system actions.

## Installation

**Note**: This project uses `uv` for Python package management.

```bash
git clone https://github.com/ejhusom/supervisor
cd supervisor
pip install -r requirements.txt

# Optional: For semantic search
pip install sentence-transformers
```

## Quick Start

```bash
# Interactive mode
uv run python3 main.py

# Single task
uv run python3 main.py "Analyze error.log for patterns"

# With preprocessing (SQLite + optional embeddings)
uv run python3 main.py --preprocess workspace/data/logs.log --embeddings

# Specific workflows
uv run python3 main.py --workflow evaluator --max-iterations 3 "Analyze logs"
uv run python3 main.py --workflow multi_stage --stages "Parse logs" "Find patterns" "Write report" "Analyze system logs"
```

## Configuration

Edit [config.toml](config.toml):

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

# Agents
agents_available = []  # Empty = all standard agents
agents_unavailable = []  # Exclude specific

# Logging and execution
log_full_messages = true
sandbox_timeout = 60
tool_call_output_max_length = 1000
```

## Architecture

### Components

**Supervisor** ([core/supervisor.py](core/supervisor.py))
- Central orchestrator with three tool categories:
  - Meta-tools: Create tools/agents, delegate tasks
  - Standard tools: File I/O, code execution, shell commands
  - Preprocessing tools: SQL queries, semantic search (when enabled)

**Agent** ([core/agent.py](core/agent.py))
- Minimal LLM wrapper: system prompt + tools
- Iterates until task complete or max iterations reached

**Registries** ([registry/](registry/))
- `ToolRegistry`: Dynamically created tools, persists to `workspace/tools/`
- `AgentRegistry`: Created agents, persists to `workspace/agents/`
- Standard agents auto-register on init: `file_manager`, `shell_worker`, `log_analyst`, `log_parser`, etc.

**Sandbox** ([core/sandbox.py](core/sandbox.py))
- Safe execution with timeout and workspace isolation at `workspace/data/`

**Preprocessor** ([core/preprocessor.py](core/preprocessor.py))
- Transforms large log files into SQLite databases with optional semantic search

### Data Flow

```
User Task → Workflow → Supervisor → [Create Tools/Agents] → Delegate → Results
                                  ↓
                             Preprocessing Tools
                             (SQLite, Embeddings)
```

## Workflows

**Simple**: Direct passthrough to supervisor (default).

**Evaluator**: Runs task, evaluates result, retries with feedback if needed.
```python
workflow = EvaluatorWorkflow(supervisor, max_iterations=3)
result = workflow.run("Find error patterns")
```

**Multi-Stage**: Sequential stages with context accumulation.
```python
workflow = MultiStageWorkflow(supervisor, stages=[
    "Parse log entries",
    "Identify error patterns", 
    "Generate summary report"
])
result = workflow.run("Analyze system logs")
```

## Preprocessing

For large log files, enable preprocessing to create queryable databases:

```bash
uv run python3 main.py --preprocess logs/app.log --embeddings
```

**Available tools after preprocessing**:
- `query_logs_from_sqlite_database(sql)` - SQL queries
- `get_error_logs_from_sqlite_database(limit)` - Get errors
- `search_logs_from_sqlite_database(keyword)` - Keyword search
- `search_similar_from_embeddings(query)` - Semantic search (requires `--embeddings`)

**Programmatic usage**:
```python
from core.preprocessor import Preprocessor, SQLiteLogIngestion, EmbeddingRAG

preprocessor = Preprocessor()
preprocessor.add_step(SQLiteLogIngestion())
preprocessor.add_step(EmbeddingRAG(chunk_size=100))
preprocessor.process(Path("logs/app.log"))
```

## Evaluation

Systematic testing of configurations on log analysis tasks:

```bash
# Generate ground truth
uv run python3 eval/eval_ground_truth.py logs/sample.log > ground_truth.json

# Run experiments with multiple configurations
uv run python3 eval/eval_experiment.py \
  --log logs/sample.log \
  --ground-truth ground_truth.json \
  --experiments experiments.json \
  --output results.csv

# Analyze results
uv run python3 eval/analyze_results.py results.csv
```

## API Usage

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

# Execute task
workflow = EvaluatorWorkflow(supervisor, max_iterations=3)
result = workflow.run("Analyze logs for security issues")
print(result["content"])
```

## API Server

FastAPI server for multi-session task execution:

```bash
uv run uvicorn api.server:app --host 0.0.0.0 --port 8000
# Or: uv run python3 -m api.server
```

Key endpoints: `/sessions`, `/sessions/{id}/tasks`, `/jobs/{job_id}/status`

## Project Structure

```
supervisor/
├── core/                  # Core framework
│   ├── agent.py          # Agent implementation
│   ├── supervisor.py     # Supervisor orchestration
│   ├── llm_client.py     # LLM interface
│   ├── sandbox.py        # Code execution
│   ├── preprocessor.py   # Log preprocessing
│   └── standard_tools.py # Built-in tools
├── workflows/            # Workflow implementations
├── registry/             # Dynamic registries
├── eval/                 # Evaluation framework
├── api/                  # FastAPI server
├── instructions/         # Agent guidance
├── workspace/            # Runtime workspace
│   ├── data/            # User data
│   ├── tools/           # Generated tools
│   └── agents/          # Generated agents
├── logs/                 # Execution logs
└── config.toml          # Configuration
```

## Key Patterns

**Creating Tools Dynamically**:
Tools created via `create_tool` persist to `workspace/tools/`. Test with `execute_python` first, then register.

**Creating Specialized Agents**:
Agents are spawned for focused subtasks with specific tool subsets:
```python
{
  "name": "log_analyzer",
  "system_prompt": "You analyze log patterns...",
  "tools": ["query_logs_from_sqlite_database", "write_file"]
}
```

**Multi-agent Delegation**:
```python
supervisor.run("""
1. Use log_parser to extract template from the log line
2. Use log_template_critic to validate
3. Use log_template_refiner to produce final template
""")
```

## Testing

```bash
uv run pytest tests/
uv run python3 tests/test_workspace.py
uv run python3 tests/test_workflows.py
```

## Streamlit UI

```bash
streamlit run app.py
```

Features: Configure LLM settings, upload files, run queries, view execution logs.

## Package Management

Uses `uv` for fast Python package management:
```bash
uv add package-name              # Add dependency
uv run python3 script.py         # Run scripts
```

## Acknowledgments

Part of the INTEND project for intent-based system management.
