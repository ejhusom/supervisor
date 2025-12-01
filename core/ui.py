"""
UI helper for consistent terminal output across iExplain.
Provides cohesive visual styling matching PrintManager.
"""

from .config import config


class UI:
    """Simple UI helper for framework-level messages, matching PrintManager style."""
    
    def __init__(self):
        self.use_colors = config.get("print_use_colors", True)
        self.colors = self._init_colors()
    
    def _init_colors(self):
        """Initialize color codes (matching PrintManager)."""
        if not self.use_colors:
            return {k: '' for k in ['cyan', 'green', 'yellow', 'blue', 'bold', 'red', 'end']}
        
        return {
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
    
    def header(self, text: str):
        """Print a box-style header (matching agent_start style)."""
        c = self.colors
        print(f"\n{c['bold']}{c['cyan']}╭─ {text} {'─' * (65 - len(text))}╮{c['end']}")
    
    def header_end(self):
        """Close a header box."""
        c = self.colors
        print(f"{c['bold']}{c['cyan']}╰{'─' * 68}╯{c['end']}\n")
    
    def section(self, text: str):
        """Print a section with emoji marker."""
        c = self.colors
        print(f"\n{c['bold']}{c['cyan']}🔧 {text}{c['end']}")
    
    def info(self, text: str):
        """Print info line with checkmark."""
        c = self.colors
        print(f"{c['green']}  ✓{c['end']} {text}")
    
    def detail(self, label: str, value: str):
        """Print detail with tree structure."""
        print(f"    └─ {label}: {value}")
    
    def success(self, text: str):
        """Print success message."""
        c = self.colors
        print(f"\n{c['green']}✓ {c['bold']}{text}{c['end']}\n")
    
    def task(self, text: str):
        """Print task description with emoji."""
        c = self.colors
        print(f"\n{c['bold']}{c['blue']}📋 Task:{c['end']} {text}")
    
    def separator(self):
        """Print a separator line."""
        print(f"{'─' * 70}")
    
    def result_header(self):
        """Print result section header."""
        c = self.colors
        print(f"\n{c['bold']}{c['cyan']}╭─ RESULT {'─' * 60}╮{c['end']}")
    
    def result_end(self):
        """Close result section."""
        c = self.colors
        print(f"{c['bold']}{c['cyan']}╰{'─' * 68}╯{c['end']}\n")
    
    def workflow_info(self, workflow_type: str, description: str):
        """Print workflow information."""
        c = self.colors
        print(f"{c['green']}  ✓{c['end']} {description}")
    
    def error(self, text: str):
        """Print error message."""
        c = self.colors
        print(f"{c['red']}✗ {text}{c['end']}")


# Singleton instance
_ui = None

def get_ui():
    """Get or create the global UI instance."""
    global _ui
    if _ui is None:
        _ui = UI()
    return _ui