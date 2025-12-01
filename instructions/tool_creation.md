# Tool Creation Guide

Guidelines for creating high-quality, reusable tools.

## Tool Structure

A tool is a Python function with:
- Clear purpose
- Type hints
- Docstring
- Error handling
- Return value

```python
def tool_name(param: type) -> return_type:
    """
    Brief description.
    
    Args:
        param: What it is
    
    Returns:
        What gets returned
    """
    try:
        # Implementation
        result = do_something(param)
        return result
    except Exception as e:
        return {"error": str(e)}
```

## Testing Pattern

Always test before creating:

```python
# 1. Write tool code
code = """
def process_data(text: str) -> dict:
    lines = text.split('\\n')
    return {"line_count": len(lines)}
"""

# 2. Test with execute_code, and assign the result to `result
execute_code(code + """
result = process_data("line1\\nline2\\nline3")
print(result)  # Should show {"line_count": 3}
""")

# 3. Create if successful
create_tool(
    name="process_data",
    code=code,
    description="Counts lines in text",
    parameters_schema={...}
)
```

## Common Patterns

### File Reading
```python
def read_file(filepath: str) -> str:
    """Read file contents."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {filepath}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Parsing
```python
def parse_log_line(line: str) -> dict:
    """Parse structured log line."""
    import re
    pattern = r'(?P<timestamp>\S+)\s+(?P<level>\w+)\s+(?P<message>.*)'
    match = re.match(pattern, line)
    return match.groupdict() if match else {}
```

### Data Processing
```python
def filter_errors(logs: list) -> list:
    """Extract error entries from logs."""
    return [log for log in logs if log.get('level') == 'ERROR']
```

## Parameter Schema

JSON schema defines the function signature:

```python
parameters_schema = {
    "type": "object",
    "properties": {
        "filepath": {
            "type": "string",
            "description": "Path to file"
        },
        "encoding": {
            "type": "string",
            "description": "File encoding (default: utf-8)",
            "default": "utf-8"
        }
    },
    "required": ["filepath"]
}
```

## Best Practices

**Do:**
- Keep tools small and focused
- Use standard library when possible
- Return structured data (dict/list)
- Handle errors gracefully
- Add helpful docstrings

**Don't:**
- Make tools that do too much
- Use global state
- Assume file paths exist
- Ignore exceptions
- Return ambiguous types

## Examples

### Good: Focused, Clear
```python
def count_errors(log_file: str) -> int:
    """Count ERROR lines in log file."""
    count = 0
    with open(log_file, 'r') as f:
        for line in f:
            if 'ERROR' in line:
                count += 1
    return count
```

### Bad: Too Much, Unclear
```python
def analyze_logs(file, options=None):
    # Does parsing, filtering, aggregation, reporting...
    # Returns ??? format
    pass
```

## Debugging

If tool creation fails:

1. **Check syntax** - Run `execute_code` first
2. **Test with real data** - Use actual inputs
3. **Check imports** - Verify packages available
4. **Simplify** - Remove complexity, test minimal version
5. **Read error** - Error messages tell you what's wrong
