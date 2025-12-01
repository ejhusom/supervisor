"""
Standard tools that any agent can use.
These are NOT meta-tools - they don't modify the system.
They provide basic code execution and file operations.
"""

from typing import Dict, Any, List
from pathlib import Path
from .sandbox import Sandbox
from .config import config


def get_standard_tools(sandbox: Sandbox) -> Dict[str, Dict[str, Any]]:
    """
    Get standard tools that any agent can use.
    
    Args:
        sandbox: Sandbox instance for code execution
    
    Returns:
        Dict of tool definitions
    """
    
    def execute_python(code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute Python code in sandbox."""
        result = sandbox.execute(code, context)
        return {
            "success": result["success"],
            "output": result["output"],
            "error": result["error"],
            "return_value": result.get("return_value")
        }
    
    def run_command(command: str, args: List[str], input_data: str = None) -> Dict[str, Any]:
        """Execute a Unix command."""
        return sandbox.execute_command(
            command=command,
            args=args,
            input_data=input_data
        )
    
    def run_shell(command_line: str) -> Dict[str, Any]:
        """Execute a shell command line with pipes."""
        return sandbox.execute_shell(command_line)
    
    def read_file(filepath: str) -> str:
        """Read a file from the workspace."""
        try:
            full_path = Path(sandbox.workspace) / filepath
            # Security: Resolve and check path is within workspace
            resolved_path = full_path.resolve()
            workspace_resolved = Path(sandbox.workspace).resolve()
            
            try:
                resolved_path.relative_to(workspace_resolved)
            except ValueError:
                return f"Error: Access denied - path outside workspace: {filepath}"
            
            with open(resolved_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found: {filepath}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(filepath: str, content: str) -> str:
        """Write content to a file in the workspace."""
        try:
            full_path = Path(sandbox.workspace) / filepath
            # Security: Resolve and check path is within workspace
            resolved_path = full_path.resolve()
            workspace_resolved = Path(sandbox.workspace).resolve()
            
            try:
                resolved_path.relative_to(workspace_resolved)
            except ValueError:
                return f"Error: Access denied - path outside workspace: {filepath}"
            
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {filepath}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def list_files(directory: str = ".") -> List[str]:
        """List files in a directory within the workspace."""
        try:
            full_path = Path(sandbox.workspace) / directory
            if not full_path.exists():
                return [f"Error: Directory not found: {directory}"]
            
            files = []
            for item in full_path.iterdir():
                prefix = "[DIR] " if item.is_dir() else "[FILE]"
                files.append(f"{prefix} {item.name}")
            return files
        except Exception as e:
            return [f"Error listing files: {str(e)}"]
    
    def pwd() -> str:
        """Get current working directory."""
        return str(sandbox.workspace)
    
    # Tool definitions with schemas
    return {
        "execute_python": {
            "function": execute_python,
            "schema": {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code in a safe sandbox. Returns output and any result assigned to 'result' variable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Assign final result to 'result' variable if you want to capture it."
                            },
                            "context": {
                                "type": "object",
                                "description": "Optional variables to make available as __iexplain__context__ dict"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        },
        "run_command": {
            "function": run_command,
            "schema": {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a Unix command safely (grep, awk, sed, find, ls, cat, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command name (e.g., 'grep', 'find', 'cat')"
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Command arguments as list of strings"
                            },
                            "input_data": {
                                "type": "string",
                                "description": "Optional stdin data to pipe to command"
                            }
                        },
                        "required": ["command", "args"]
                    }
                }
            }
        },
        "run_shell": {
            "function": run_shell,
            "schema": {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Execute a shell command line with pipes and redirects (e.g., 'grep ERROR log.txt | wc -l').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command_line": {
                                "type": "string",
                                "description": "Complete shell command with pipes/redirects"
                            }
                        },
                        "required": ["command_line"]
                    }
                }
            }
        },
        "read_file": {
            "function": read_file,
            "schema": {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file from the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to file relative to workspace (e.g., 'data.txt' or 'logs/app.log')"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            }
        },
        "write_file": {
            "function": write_file,
            "schema": {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to file relative to workspace"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            }
                        },
                        "required": ["filepath", "content"]
                    }
                }
            }
        },
        "list_files": {
            "function": list_files,
            "schema": {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to list (relative to workspace, default: '.')",
                                "default": "."
                            }
                        }
                    }
                }
            }
        },
        "pwd": {
            "function": pwd,
            "schema": {
                "type": "function",
                "function": {
                    "name": "pwd",
                    "description": "Get the current working directory path.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        }
    }
