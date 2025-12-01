"""
Formatted terminal output for agent execution.
"""

import json
from typing import Any, Dict, List


class PrintManager:
    """Handles clean, formatted terminal output during agent execution."""
    
    def __init__(
        self,
        verbosity: str = 'normal',
        use_colors: bool = True,
        truncate_length: int = 200
    ):
        """
        Args:
            verbosity: 'quiet', 'normal', or 'verbose'
            use_colors: Enable ANSI color codes
            truncate_length: Max length for truncated output
        """
        self.verbosity = verbosity
        self.use_colors = use_colors
        self.truncate_length = truncate_length
        
        # Color codes
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m'
        } if use_colors else {k: '' for k in [
            'header', 'blue', 'cyan', 'green', 'yellow', 'red', 'bold', 'underline', 'end'
        ]}
    
    def agent_start(self, agent_name: str, parent: str = None):
        """Print when agent starts."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        parent_text = f" (called by {parent})" if parent else ""
        print(f"\n{c['bold']}{c['cyan']}╭─ Agent: {agent_name}{parent_text} {'─' * (50 - len(agent_name) - len(parent_text))}╮{c['end']}")
    
    def iteration_header(self, iteration: int, max_iterations: int, agent_name: str = None):
        """Print iteration header."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        agent_text = f" ({agent_name})" if agent_name else ""
        print(f"{c['bold']}│ Iteration {iteration}/{max_iterations}{agent_text}{c['end']}")
    
    def tool_call(self, tool_name: str, arguments: Dict):
        """Format tool call with key arguments."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        print(f"├{'─' * 60}┤")
        print(f"{c['yellow']}│ 📞 Tool: {tool_name}{c['end']}")
        
        # Show key arguments (truncate if needed)
        if arguments:
            args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
            for key, value in args_dict.items():
                value_str = str(value)
                if len(value_str) > self.truncate_length:
                    value_str = value_str[:self.truncate_length] + "..."
                print(f"│    └─ {key}: {repr(value_str)}")
    
    def tool_result(self, tool_name: str, result: Any):
        """Summarize tool result."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        
        # Summarize result
        if isinstance(result, (list, tuple)):
            summary = f"Returned {len(result)} items"
        elif isinstance(result, dict):
            if 'error' in result:
                summary = f"{c['red']}Error: {result['error']}{c['end']}"
            else:
                summary = f"Dict with {len(result)} keys: {', '.join(list(result.keys())[:3])}"
        elif isinstance(result, str):
            if len(result) > self.truncate_length:
                summary = result[:self.truncate_length] + "..."
            else:
                summary = result
        else:
            summary = str(result)[:self.truncate_length]
        
        print(f"{c['green']}│    └─ Result: {summary}{c['end']}")
    
    def response_preview(self, content: str):
        """Show response preview."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        print(f"├{'─' * 60}┤")
        
        if len(content) > self.truncate_length:
            preview = content[:self.truncate_length] + "..."
        else:
            preview = content
        
        print(f"{c['cyan']}│ ✓ Response: {preview}{c['end']}")
    
    def agent_end(self):
        """Print when agent ends."""
        if self.verbosity == 'quiet':
            return
        
        c = self.colors
        print(f"{c['bold']}{c['cyan']}╰{'─' * 60}╯{c['end']}")
    
    def separator(self):
        """Print a separator line."""
        if self.verbosity == 'quiet':
            return
        print()


class QuietPrinter(PrintManager):
    """PrintManager that prints nothing (for background operations)."""
    
    def __init__(self):
        super().__init__(verbosity='quiet')
