# API Reference

## Core Classes

### LLMClient

Interface to LLM providers.

```python
from core.llm_client import LLMClient

client = LLMClient(
    provider="anthropic",  # or "openai", "ollama"
    model="claude-sonnet-4-20250514",
    api_key="your-key"  # optional, reads from env
)

response = client.complete(
    messages=[{"role": "user", "content": "Hello"}],
    system="You are a helpful assistant",
    tools=None,  # optional tool schemas
    max_tokens=16384,
    temperature=0.0
)
# Returns: {content, tool_calls, usage, model, finish_reason}
```

### Supervisor

Main orchestration component.

```python
from core.supervisor import Supervisor
from core.llm_client import LLMClient
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

supervisor = Supervisor(
    llm_client=LLMClient(...),
    tool_registry=ToolRegistry(),
    agent_registry=AgentRegistry(),
    instructions_dir="instructions",
    preprocessor=None  # optional
)

result = supervisor.run(
    message="Analyze logs for errors",
    context={"key": "value"}  # optional
)
# Returns: {content, tool_calls, history}
```

### Agent

LLM + tools + system prompt.

```python
from core.agent import Agent

agent = Agent(
    name="analyzer",
    system_prompt="You analyze logs for patterns...",
    llm_client=llm_client,
    tools={"tool_name": {"function": fn, "schema": {...}}},
    temperature=0.0,
    max_tokens=8192
)

result = agent.run(
    message="Find errors",
    context={"data": "..."},
    max_iterations=20,
    parent_agent="supervisor"  # optional
)
# Returns: {content, tool_calls, history}
```

### Sandbox

Safe code execution.

```python
from core.sandbox import Sandbox

sandbox = Sandbox(
    timeout=60,
    workspace="./workspace/data"
)

# Python execution
result = sandbox.execute(
    code="result = 2 + 2",
    context={"x": 10}  # optional
)
# Returns: {success, output, error, return_value}

# Unix command
result = sandbox.execute_command(
    command="grep",
    args=["-c", "ERROR", "app.log"]
)
# Returns: {success, output, error, returncode}

# Shell pipeline
result = sandbox.execute_shell(
    command_line="grep ERROR app.log | wc -l"
)
```

### Preprocessor

Log preprocessing system.

```python
from pathlib import Path
from core.preprocessor import (
    Preprocessor, 
    SQLiteLogIngestion, 
    EmbeddingRAG
)

preprocessor = Preprocessor(workspace=Path("workspace/data"))

# Add steps
preprocessor.add_step(SQLiteLogIngestion(parser_name="auto"))
preprocessor.add_step(EmbeddingRAG(
    chunk_size=100,
    model_name='all-MiniLM-L6-v2',
    overlap=10,
    max_chunks=None  # optional limit
))

# Process
metadata = preprocessor.process(Path("logs/app.log"))

# Get tools
tools = preprocessor.get_all_tools()
```

## Registries

### ToolRegistry

Dynamic tool storage.

```python
from registry.tool_registry import ToolRegistry

registry = ToolRegistry(persist_dir="workspace/tools")

# Register tool
registry.register(
    name="my_tool",
    function=my_function,
    schema={
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "...",
            "parameters": {...}
        }
    },
    code="def my_tool(x): return x * 2"  # optional
)

# Get tool
tool = registry.get("my_tool")
# Returns: {function, schema, code}

# List tools
tools = registry.list_tools()  # Returns: ["tool1", "tool2"]

# Get all tools
all_tools = registry.get_all()  # Returns: {"name": {function, schema, code}}
```

### AgentRegistry

Dynamic agent storage.

```python
from registry.agent_registry import AgentRegistry

registry = AgentRegistry(persist_dir="workspace/agents")

# Register agent
registry.register(
    name="analyzer",
    agent=agent_instance,
    config={
        "system_prompt": "...",
        "tools": ["tool1", "tool2"]
    }
)

# Get agent
agent = registry.get("analyzer")

# List agents
agents = registry.list_agents()  # Returns: ["agent1", "agent2"]
```

## Workflows

### SimpleWorkflow

```python
from workflows import SimpleWorkflow

workflow = SimpleWorkflow(supervisor)
result = workflow.run(task="...", context={})
```

### EvaluatorWorkflow

```python
from workflows import EvaluatorWorkflow

workflow = EvaluatorWorkflow(
    supervisor,
    max_iterations=3,
    verbose=True
)
result = workflow.run(task="...", context={})
# Returns: {..., evaluation_history, iterations_used}
```

### MultiStageWorkflow

```python
from workflows import MultiStageWorkflow

workflow = MultiStageWorkflow(
    supervisor,
    stages=["Stage 1", "Stage 2", "Stage 3"],
    accumulate_context=True,
    verbose=True
)
result = workflow.run(task="...", context={})
# Returns: {..., stage_results, stages_completed}
```

### Predefined Workflows

```python
from workflows import PredefinedMultiStageWorkflow

# Analysis workflow
workflow = PredefinedMultiStageWorkflow.analysis_workflow(supervisor)

# Research workflow
workflow = PredefinedMultiStageWorkflow.research_workflow(supervisor)

# Transformation workflow
workflow = PredefinedMultiStageWorkflow.transformation_workflow(supervisor)
```

## Utilities

### Config

```python
from core.config import config

# Access settings
provider = config.get("provider")
workspace = config.get("workspace")

# All available keys:
# provider, model, temperature, max_tokens
# workspace, workspace_data, workspace_tools, workspace_agents
# log_dir, logging_enabled, log_full_messages
# sandbox_timeout, tool_call_output_max_length
# embeddings_chunk_size
# print_verbosity, print_use_colors, print_truncate_length
# tools_available, tools_unavailable
```

### Logger

```python
from core.logger import get_logger

logger = get_logger()

# Start session
session_id = logger.start_session(
    task="Analyze logs",
    config={...}
)

# Log agent start
idx = logger.log_agent_start(
    agent_name="analyzer",
    message="Find errors",
    context={...},
    parent_agent=None
)

# Log iteration
logger.log_iteration(
    interaction_idx=idx,
    iteration=1,
    response_content="...",
    tool_calls=[...],
    model_info={...},
    tool_call_results=[...]
)

# Log agent end
logger.log_agent_end(
    interaction_idx=idx,
    result="...",
    total_tool_calls=5
)

# End session
log_file = logger.end_session(final_result="...")
```

### UI

```python
from core.ui import get_ui

ui = get_ui()

ui.header("Section Title")
ui.section("Subsection")
ui.info("Information message")
ui.detail("Label", "value")
ui.success("Success message")
ui.error("Error message")
ui.separator()
ui.header_end()
```

## Standard Tools

Available via `get_standard_tools(sandbox)`:

```python
from core.standard_tools import get_standard_tools
from core.sandbox import Sandbox

tools = get_standard_tools(Sandbox())

# Tools returned:
{
    "execute_python": {
        "function": callable,
        "schema": {...}
    },
    "run_command": {...},
    "run_shell": {...},
    "read_file": {...},
    "write_file": {...},
    "list_files": {...},
    "pwd": {...}
}
```

## Preprocessing Tools

Available after `preprocessor.process()`:

### SQL Tools

```python
# Query logs
query_logs_from_sqlite_database(
    sql="SELECT * FROM logs WHERE level = ?",
    params=["ERROR"]
)

# Get errors
get_error_logs_from_sqlite_database(limit=100)

# Search
search_logs_from_sqlite_database(keyword="timeout", limit=50)

# Statistics
get_log_stats_from_sqlite_database()
```

### Embedding Tools

```python
# Semantic search
search_similar_from_embeddings(
    query="authentication failures",
    top_k=5,
    min_similarity=0.0
)

# Find similar chunks
find_similar_to_chunk_from_embeddings(chunk_index=42, top_k=5)

# Get chunk
get_chunk_by_index_from_embeddings(chunk_index=42)

# Get info
get_embedding_info()
```

## Error Handling

All functions return dicts with success/error information:

```python
# Successful execution
{
    "success": True,
    "output": "...",
    "return_value": ...
}

# Failed execution
{
    "success": False,
    "error": "Error message",
    "output": ""
}
```

Exception handling:

```python
try:
    result = supervisor.run(task)
except Exception as e:
    print(f"Error: {e}")
    # Handle error
```

## Type Hints

```python
from typing import Dict, Any, List, Optional

def my_tool(x: int) -> Dict[str, Any]:
    return {"result": x * 2}

def my_function(
    required_param: str,
    optional_param: Optional[int] = None
) -> List[Dict[str, Any]]:
    return [{"key": "value"}]
```

## Common Patterns

### Initialize System

```python
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
supervisor = Supervisor(
    llm_client=llm,
    tool_registry=ToolRegistry(),
    agent_registry=AgentRegistry(),
    instructions_dir="instructions"
)
```

### Run with Workflow

```python
from workflows import EvaluatorWorkflow

workflow = EvaluatorWorkflow(supervisor, max_iterations=3)
result = workflow.run("Analyze logs for security issues")
print(result["content"])
```

### Add Preprocessing

```python
from core.preprocessor import Preprocessor, SQLiteLogIngestion

preprocessor = Preprocessor()
preprocessor.add_step(SQLiteLogIngestion())
preprocessor.process(Path("logs/app.log"))

supervisor = Supervisor(
    llm_client=llm,
    tool_registry=ToolRegistry(),
    agent_registry=AgentRegistry(),
    instructions_dir="instructions",
    preprocessor=preprocessor
)
```

### Custom Tool

```python
def my_tool(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

schema = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"}
            },
            "required": ["x", "y"]
        }
    }
}

tool_registry.register("my_tool", my_tool, schema)
```
