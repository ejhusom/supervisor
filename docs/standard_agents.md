# Standard Agents Configuration

Standard agents are pre-configured agents automatically registered when the Supervisor initializes. They provide common functionality without requiring runtime agent creation.

## Available Standard Agents

### General Purpose Agents

- **`file_manager`**: Reads, writes, and lists files within the workspace sandbox
  - Tools: `read_file`, `write_file`, `list_files`

- **`shell_worker`**: Executes whitelisted Unix commands and shell pipelines
  - Tools: `run_command`, `run_shell`

- **`log_analyst`**: Analyzes logs using SQL queries and embeddings (requires preprocessing)
  - Tools: `query_logs_from_sqlite_database`, `get_error_logs_from_sqlite_database`, `search_similar_from_embeddings`
  - Note: Only registered when preprocessing is enabled

### Log Analysis Pipeline Agents

These agents are adapted from the agent-green research project and implement a multi-agent log analysis pipeline:

- **`log_parser`**: Extracts structured templates from raw log messages
  - Replaces variable values (IPs, IDs, numbers) with `<*>` placeholders
  - Focuses on message body, ignoring headers (timestamp, level, class)
  - Tools: `read_file`, `write_file`

- **`log_template_critic`**: Validates log templates for correctness
  - Checks if all variables are properly abstracted as `<*>`
  - Verifies constant text preservation
  - Returns corrected template if issues found
  - Tools: `read_file`, `write_file`

- **`log_template_refiner`**: Refines templates by comparing parser and critic outputs
  - Merges or selects best template
  - Prefers critic template when uncertain
  - Regenerates from original if both are incorrect
  - Tools: `read_file`, `write_file`

- **`log_anomaly_detector`**: Detects anomalies in log sessions
  - Analyzes textual anomalies (errors, exceptions, crashes)
  - Detects behavioral anomalies (unusual sequences, missing events)
  - Outputs binary decision: 0 (normal) or 1 (anomalous)
  - Tools: `read_file`, `query_logs_from_sqlite_database`

- **`log_preprocessor`**: Prepares log sessions for analysis
  - Extracts message bodies from raw logs
  - Removes headers while preserving sequence
  - Structures data for downstream analysis
  - Tools: `read_file`, `write_file`, `run_command`

- **`anomaly_critic`**: Validates anomaly detection decisions
  - Checks for missed errors or false positives
  - Verifies behavioral pattern analysis
  - Returns corrected decision (0 or 1)
  - Tools: `read_file`

## Configuration

Control which standard agents are registered via `config.toml`:

```toml
# Register all standard agents (default)
agents_available = []
agents_unavailable = []

# Register only specific agents
agents_available = ["file_manager", "shell_worker"]
agents_unavailable = []

# Register all except specific agents
agents_available = []
agents_unavailable = ["log_analyst"]
```

### Configuration Behavior

- **`agents_available` (empty list)**: All standard agents are registered
- **`agents_available` (non-empty)**: Only listed agents are registered
- **`agents_unavailable`**: Excludes listed agents from registration
- If both are specified, `agents_available` is applied first, then exclusions

## Usage

### Delegating to Standard Agents

Once registered, standard agents can be used via the `delegate_to_agent` meta-tool:

```python
# Supervisor delegates a file operation to the file_manager agent
supervisor.run("Use file_manager to read config.toml and summarize its contents")
```

### Checking Registered Agents

```python
# List all registered agents (including standard and dynamically created)
registered = agent_registry.list_agents()
print(f"Available agents: {registered}")
```

### Persistence

Standard agents are persisted to `workspace/agents/` as JSON files (e.g., `file_manager.json`). They are recreated on Supervisor initialization based on configuration.

## Multi-Agent Workflows

The log analysis agents are designed to work together in pipelines. Common patterns:

### Log Parsing Pipeline

```
log_parser → log_template_critic → log_template_refiner
```

1. **Parse**: Extract initial template from raw log
2. **Critique**: Validate and correct template
3. **Refine**: Produce final template from parser + critic outputs

Example delegation:
```python
# Supervisor delegates sequential tasks
supervisor.run("""
1. Use log_parser to extract template from: "081109 204005 INFO: Receiving block blk_123 from 10.0.0.1:8080"
2. Use log_template_critic to validate the template
3. Use log_template_refiner to produce final template
4. Write result to templates.txt
""")
```

### Anomaly Detection Pipeline

```
log_preprocessor → log_anomaly_detector → anomaly_critic
```

1. **Preprocess**: Clean logs, extract message bodies
2. **Detect**: Analyze session for anomalies (textual + behavioral)
3. **Critique**: Validate detection decision

Example delegation:
```python
supervisor.run("""
1. Use log_preprocessor to clean session logs from session_001.log
2. Use log_anomaly_detector to determine if session is anomalous
3. Use anomaly_critic to validate the decision
4. Write final result (0 or 1) to result.txt
""")
```

### Combined Analysis

```
log_analyst (SQL) → log_anomaly_detector → anomaly_critic
```

For large log databases with preprocessing:
```python
supervisor.run("""
1. Use log_analyst to query error logs for session_id = 'xyz'
2. Use log_anomaly_detector to analyze the error sequence
3. Use anomaly_critic to confirm the anomaly classification
""")
```

## Adding New Standard Agents

Edit `core/standard_agents.py` and add to `STANDARD_AGENT_SPECS`:

```python
STANDARD_AGENT_SPECS: List[Dict[str, Any]] = [
    {
        "name": "my_custom_agent",
        "system_prompt": "You are a specialized agent that...",
        "tools": ["tool1", "tool2"],
    },
    # ... existing agents
]
```

The agent will automatically be:
- Filtered by configuration
- Registered in `AgentRegistry`
- Persisted to `workspace/agents/my_custom_agent.json`
- Available for delegation

## Implementation Details

- Standard agents are registered in `Supervisor.__init__` via `_register_standard_agents()`
- Tool availability is checked - agents with zero available tools are skipped
- Preprocessing-dependent agents (like `log_analyst`) only register when preprocessing tools exist
- Registration respects `tools_available`/`tools_unavailable` config for tool filtering
