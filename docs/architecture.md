# Architecture

## Overview

iExplain uses a self-modifying agent architecture where a supervisor creates tools and specialized agents at runtime to solve tasks.

## Core Components

### Supervisor

Central orchestrator with three tool categories:

**Meta-tools** (system modification):
- `create_tool` - Generate new tools from code
- `create_agent` - Spawn specialized agents
- `delegate_to_agent` - Hand off tasks
- `read_instructions` - Load guidance

**Standard tools** (execution):
- `execute_python` - Run code in sandbox
- `run_command/run_shell` - Unix commands
- `read_file/write_file` - File I/O
- `list_files/pwd` - Filesystem

**Preprocessing tools** (when enabled):
- `query_logs_from_sqlite_database` - SQL queries
- `search_similar_from_embeddings` - Semantic search
- `get_error_logs_from_sqlite_database` - Error extraction

### Agent

Minimal implementation: LLM + tools + system prompt.

```python
agent.run(message, context, max_iterations=20)
```

Iterates until:
- No tool calls (task complete)
- Max iterations reached

### Registries

**ToolRegistry**: Stores dynamically created tools
- Persists to `workspace/tools/`
- Loads on startup

**AgentRegistry**: Stores created agents
- Persists to `workspace/agents/`
- Agents recreated from config

### Sandbox

Safe execution environment:
- Python code in subprocess
- Unix commands (whitelisted)
- Timeout protection
- Workspace isolation

Working directory: `workspace/data/`

### Preprocessor

Transforms large log files into queryable formats:

**SQLiteLogIngestion**:
- Parses log lines
- Creates indexed database
- Enables SQL queries

**EmbeddingRAG**:
- Chunks logs (default: 100 lines)
- Creates embeddings (sentence-transformers)
- Enables semantic search

## Execution Flow

```
1. User provides task
2. Workflow receives task
3. Supervisor analyzes requirements
4. [Optional] Create new tools
5. [Optional] Create specialized agents
6. Execute via tool calls or delegation
7. Return results
```

## Self-Modification

Supervisor can modify itself at runtime:

**Tool Creation**:
```python
# 1. Test code
execute_python("def my_tool(x): return x * 2; result = my_tool(5)")

# 2. Create tool
create_tool(
    name="my_tool",
    code="def my_tool(x): return x * 2",
    description="Doubles a number",
    parameters_schema={...}
)
```

**Agent Creation**:
```python
create_agent(
    name="log_analyzer",
    system_prompt="You analyze logs for errors...",
    tools=["parse_log", "filter_errors", "query_logs_from_sqlite_database"]
)

delegate_to_agent("log_analyzer", "Find connection errors")
```

The tools and agents created by Supervisor persist in the workspace for use in later queries.

## Workflow Layer

Workflows wrap the supervisor to implement execution strategies:

- **SimpleWorkflow**: Direct passthrough (default)
- **EvaluatorWorkflow**: Run → evaluate → retry loop
- **MultiStageWorkflow**: Sequential stages with context accumulation

See `WORKFLOWS.md` for details.

## Data Flow

### Without Preprocessing
```
Task → Supervisor → Agent → [Tools] → Results
```

### With Preprocessing
```
Log File → Preprocessor → [SQLite + Embeddings]
                                ↓
Task → Supervisor → Agent → [Preprocessing Tools] → Results
```

## Component Interactions

```
main.py
  ├─ Workflow (orchestration strategy)
  │    └─ Supervisor (task execution)
  │         ├─ Agent (iterative reasoning)
  │         │    └─ LLMClient (API calls)
  │         ├─ ToolRegistry (dynamic tools)
  │         ├─ AgentRegistry (created agents)
  │         └─ Sandbox (code execution)
  └─ Preprocessor (optional)
       └─ [SQLite, Embeddings]
```


## Configuration

`config.toml` controls:
- LLM provider and model
- Workspace paths
- Sandbox settings
- Tool availability
- Preprocessing options

## Logging

All interactions logged to `logs/` as JSON:
- Session metadata
- Agent interactions
- Tool calls and results
- Iteration history
- Duration and token usage

Format designed for analysis and visualization.
