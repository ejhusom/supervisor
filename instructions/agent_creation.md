# Agent Creation Guide

Guidelines for creating effective specialized agents.

## Agent Purpose

Agents are specialists that combine:
- Domain expertise (system prompt)
- Specific tools
- Task focus

Create agents when tasks require:
- Multiple tool calls
- Domain knowledge
- Complex reasoning
- Iterative refinement

## Agent Structure

```python
create_agent(
    name="specialist_name",
    system_prompt="Clear instructions with examples",
    tools=["tool1", "tool2", "tool3"]
)
```

## System Prompt Design

**Good prompts are:**
- Specific about role
- Clear about approach
- Include examples
- Define output format

**Template:**
```
You are a [role] specialist that [does what].

Your approach:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Example:
[Show a concrete example]

Output format:
[Define structure]
```

## Examples

### Log Analyzer Agent
```python
create_agent(
    name="log_analyzer",
    system_prompt="""You are a log analysis specialist.

Your approach:
1. Read log files with read_file tool
2. Parse each line with parse_log tool
3. Identify ERROR and WARNING entries
4. Extract patterns and anomalies
5. Provide structured summary

Output format:
- Total lines analyzed
- Error count and types
- Key anomalies with line numbers
- Timeline of critical events

Be thorough and cite specific line numbers.""",
    tools=["read_file", "parse_log", "filter_errors"]
)
```

### Report Generator Agent
```python
create_agent(
    name="report_writer",
    system_prompt="""You are a technical report writer.

You receive analysis results and format them into clear, 
professional reports.

Structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (sections with headers)
4. Recommendations (if applicable)

Be concise, factual, and well-organized.""",
    tools=[]  # No tools needed - just synthesis
)
```

## Tool Selection

**Principles:**
- Give minimum tools needed
- Match tools to role
- 3-5 tools is ideal
- Don't overload

**By role type:**

*Analyzers:* Read, parse, filter, aggregate tools
*Generators:* Format, template, structure tools
*Processors:* Transform, convert, clean tools

## Delegation Pattern

```python
# 1. Create agent
create_agent("specialist", prompt, tools)

# 2. Delegate task
result = delegate_to_agent(
    agent_name="specialist",
    task="Specific task description",
    context={"key": "value"}  # Optional context
)

# 3. Use result
# result["response"] contains agent's output
```

## Agent Composition

Build complex workflows by chaining agents:

```python
# Stage 1: Parser
create_agent("parser", "Extract data", ["parse_log"])
result1 = delegate_to_agent("parser", "Parse logs")

# Stage 2: Analyzer
create_agent("analyzer", "Find patterns", ["filter", "aggregate"])
result2 = delegate_to_agent("analyzer", "Analyze", context=result1)

# Stage 3: Reporter
create_agent("reporter", "Write report", [])
result3 = delegate_to_agent("reporter", "Report", context=result2)
```

## Common Patterns

### Specialist with Tools
```python
# Agent that does hands-on work
create_agent(
    name="data_processor",
    system_prompt="Process and transform data using tools",
    tools=["read_file", "parse_csv", "filter_data", "save_output"]
)
```

### Specialist without Tools
```python
# Agent that synthesizes/reasons
create_agent(
    name="synthesizer",
    system_prompt="Combine results into coherent summary",
    tools=[]  # Pure reasoning, no tool calls
)
```

## Best Practices

**Do:**
- Make agents specialized (one role)
- Write clear, specific prompts
- Include examples in prompt
- Give appropriate tools only
- Define expected output format

**Don't:**
- Create generalist agents
- Give all tools to every agent
- Write vague prompts
- Assume agent knows context
- Create duplicate agents

## Debugging

If agent isn't working well:

1. **Check prompt clarity** - Is role clear?
2. **Verify tools** - Does it have what it needs?
3. **Test task** - Is task well-defined?
4. **Review output** - What's it actually doing?
5. **Iterate** - Refine prompt based on behavior

## Example Workflow

```
Task: "Analyze OpenStack logs and write report"

# Approach 1: Single specialized agent
create_agent("log_report_agent", 
    "Analyze logs and write reports", 
    ["read_file", "parse_log", "filter_errors"])

delegate_to_agent("log_report_agent", task)

# Approach 2: Pipeline of specialists
create_agent("log_analyzer", "Analyze logs", [...])
create_agent("report_writer", "Write reports", [])

result1 = delegate_to_agent("log_analyzer", "Analyze logs")
result2 = delegate_to_agent("report_writer", "Write report", result1)
```

Choose based on task complexity.
