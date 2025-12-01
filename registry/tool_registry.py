"""
Dynamic tool registry.
Tools can be registered at runtime from generated code.
"""

import json
from typing import Dict, Any, Callable
from pathlib import Path
from core.config import config


class ToolRegistry:
    """Manages dynamically created tools."""
    
    def __init__(self, persist_dir: str = None):
        """
        Args:
            persist_dir: Directory to save tool definitions (defaults to workspace/tools)
        """
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else Path(config.get("workspace_tools"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persisted tools
        self._load_persisted_tools()
    
    def register(
        self,
        name: str,
        function: Callable,
        schema: Dict[str, Any],
        code: str = None
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            function: Callable implementing the tool
            schema: LLM function schema
            code: Source code (for persistence)
        """
        self.tools[name] = {
            "function": function,
            "schema": schema,
            "code": code
        }
        
        # Persist if code provided
        if code:
            self._persist_tool(name, schema, code)
    
    def get(self, name: str) -> Dict[str, Any]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tools."""
        return self.tools
    
    def get_schemas(self, tool_names: list = None) -> list:
        """Get schemas for specified tools (or all if None)."""
        if tool_names:
            return [self.tools[name]["schema"] for name in tool_names if name in self.tools]
        return [tool["schema"] for tool in self.tools.values()]
    
    def list_tools(self) -> list:
        """List all tool names."""
        return list(self.tools.keys())
    
    def _persist_tool(self, name: str, schema: Dict, code: str) -> None:
        """Save tool to disk."""
        tool_file = self.persist_dir / f"{name}.json"
        code_file = self.persist_dir / f"{name}.py"
        
        # Save schema
        with open(tool_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Save code
        with open(code_file, 'w') as f:
            f.write(code)
    
    def _load_persisted_tools(self) -> None:
        """Load tools from disk."""
        for tool_file in self.persist_dir.glob("*.json"):
            name = tool_file.stem
            code_file = self.persist_dir / f"{name}.py"
            
            if not code_file.exists():
                continue
            
            # Load schema
            with open(tool_file, 'r') as f:
                schema = json.load(f)
            
            # Load code
            with open(code_file, 'r') as f:
                code = f.read()
            
            # Execute code to get function
            try:
                namespace = {}
                exec(code, namespace)
                
                # Find the function (assume it matches tool name or is first function)
                func = namespace.get(name)
                if not func:
                    # Find first function
                    for obj in namespace.values():
                        if callable(obj) and hasattr(obj, '__name__') and obj.__name__ != '__builtins__':
                            func = obj
                            break
                
                if func:
                    self.tools[name] = {
                        "function": func,
                        "schema": schema,
                        "code": code
                    }
            
            except Exception as e:
                print(f"Failed to load tool {name}: {e}")
