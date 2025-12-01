#!/usr/bin/env python3
"""
Quick test to verify iExplain v2 components work.
Run this before using the full system.
"""
import os
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sandbox import Sandbox
from core.llm_client import LLMClient
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry


def test_sandbox():
    """Test code execution sandbox."""
    sandbox = Sandbox()
    # Test simple execution
    result = sandbox.execute("""
result = 2 + 2
""")
    assert result["success"], "Sandbox execution failed"
    assert result["return_value"] == 4, "Sandbox returned wrong value"


def test_registries():
    """Test tool and agent registries."""
    # Test tool registry
    tools = ToolRegistry(persist_dir="/tmp/test-tools")
    def dummy_tool(x: int) -> int:
        return x * 2
    schema = {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"}
                },
                "required": ["x"]
            }
        }
    }
    tools.register("dummy_tool", dummy_tool, schema, "def dummy_tool(x): return x * 2")
    assert "dummy_tool" in tools.list_tools(), "Tool not registered"
    # Test agent registry
    agents = AgentRegistry(persist_dir="/tmp/test-agents")
    agents.register("test_agent", None, {"system_prompt": "test"})
    assert "test_agent" in agents.list_agents(), "Agent not registered"

@pytest.mark.skipif(os.getenv("ANTHROPIC_API_KEY") is None, reason="No API key set")
def test_llm_client():
    """Test LLM client initialization."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = LLMClient()

if __name__ == "__main__":
    pytest.main([__file__])