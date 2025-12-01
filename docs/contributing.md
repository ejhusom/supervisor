# Contributing

## Overview

iExplain is a research framework for agentic AI and log analysis. Contributions are welcome.

## Development Setup

```bash
git clone <repository-url>
cd iexplain
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if exists

# Run tests
pytest tests/
```

## Code Style

Follow existing patterns:

**Concise**: Keep code minimal and readable
**Simple**: Prefer simple solutions over complex ones
**Documented**: Add docstrings for public functions

**Example**:
```python
def my_function(x: int) -> int:
    """
    Brief description.
    
    Args:
        x: Input value
        
    Returns:
        Processed value
    """
    return x * 2
```

## Project Structure

```
core/           # Framework components
workflows/      # Execution strategies
registry/       # Dynamic registries
eval/           # Evaluation framework
instructions/   # Agent guidance
tests/          # Test suite
```

## Adding Features

### New Workflow

1. Create file in `workflows/`
2. Extend `Workflow` base class
3. Implement `run()` method
4. Add to `workflows/__init__.py`
5. Document in `docs/WORKFLOWS.md`

```python
from workflows.base import Workflow

class MyWorkflow(Workflow):
    def run(self, task, context=None):
        # Your logic
        return self.supervisor.run(task, context)
```

### New Preprocessing Step

1. Extend `PreprocessorStep` in `core/preprocessor.py`
2. Implement `process()` and `get_tools()`
3. Document in `docs/PREPROCESSING.md`

```python
class MyPreprocessor(PreprocessorStep):
    def process(self, input_path, workspace):
        # Process data
        return {"metadata": "..."}
    
    def get_tools(self):
        return {"tool_name": {"function": fn, "schema": {...}}}
```

### New Tool

Tools are created at runtime by the supervisor. To add a built-in tool:

1. Add to `core/standard_tools.py`
2. Include in `get_standard_tools()` return dict
3. Document in `docs/API.md`

### New Evaluation Question Type

1. Add comparison logic in `eval/eval_comparator.py`
2. Update `generate_evaluation_questions()` in `eval/eval_ground_truth.py`
3. Document in `docs/EVALUATION.md`

## Testing

### Run Tests

```bash
# All tests
pytest tests/

# Specific test
python tests/test_workflows.py

# With coverage
pytest --cov=core tests/
```

### Add Tests

Create test file in `tests/`:

```python
def test_my_feature():
    # Arrange
    setup = ...
    
    # Act
    result = my_function(setup)
    
    # Assert
    assert result == expected
```

## Documentation

### Update Docs

When adding features:

1. Update relevant doc in `docs/`
2. Update README.md if needed
3. Keep docs concise

### Documentation Style

- **Concise**: Get to the point
- **Examples**: Show concrete usage
- **Structure**: Use clear headers
- **Code**: Include runnable examples

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Update docs
6. Submit PR

**PR Description**:
- What: Brief description
- Why: Motivation
- How: Implementation approach
- Testing: How tested

## Research Contributions

For research papers or experiments:

1. Add experiments to `experiments/`
2. Document methodology
3. Include results
4. Submit findings

## Issue Guidelines

**Bug Reports**:
- Description
- Steps to reproduce
- Expected vs actual behavior
- Environment (Python version, OS, etc.)

**Feature Requests**:
- Use case
- Proposed solution
- Alternatives considered

## Code Review

Reviewers check:
- Code quality
- Test coverage
- Documentation
- Consistency with project style

## Questions?

Open an issue or discussion for:
- Implementation questions
- Design decisions
- Feature proposals
- Bug reports

## License

By contributing, you agree that your contributions will be licensed under the project license.
