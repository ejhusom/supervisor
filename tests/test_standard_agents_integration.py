#!/usr/bin/env python3
"""Test standard agent registration and configuration filtering."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config
from core.standard_tools import get_standard_tools
from core.sandbox import Sandbox
from registry.agent_registry import AgentRegistry


class MockLLMClient:
    """Mock LLM client for testing."""
    def __init__(self, provider="mock", model="mock"):
        self.provider = provider
        self.model = model


def test_config_filtering():
    """Test that config-based filtering works via Supervisor._register_standard_agents logic."""
    
    print("=" * 60)
    print("Testing Standard Agent Configuration Filtering")
    print("=" * 60)
    
    # Import after setting up path
    from core.supervisor import Supervisor
    
    # Save original config
    original_available = config.get("agents_available", [])
    original_unavailable = config.get("agents_unavailable", [])
    
    tests = [
        {
            "name": "All agents (default)",
            "agents_available": [],
            "agents_unavailable": [],
            "expected": ["file_manager", "shell_worker"],  # log_analyst needs preprocessing
        },
        {
            "name": "Only file_manager",
            "agents_available": ["file_manager"],
            "agents_unavailable": [],
            "expected": ["file_manager"],
        },
        {
            "name": "Exclude shell_worker",
            "agents_available": [],
            "agents_unavailable": ["shell_worker"],
            "expected": ["file_manager"],
        },
        {
            "name": "Only shell_worker",
            "agents_available": ["shell_worker"],
            "agents_unavailable": [],
            "expected": ["shell_worker"],
        },
    ]
    
    for test in tests:
        print(f"\nTest: {test['name']}")
        print(f"  Config: available={test['agents_available']}, unavailable={test['agents_unavailable']}")
        
        # Set config
        config["agents_available"] = test["agents_available"]
        config["agents_unavailable"] = test["agents_unavailable"]
        
        # Create fresh registries
        from registry.tool_registry import ToolRegistry
        tool_reg = ToolRegistry()
        agent_reg = AgentRegistry()
        
        # Create supervisor (which registers standard agents)
        mock_llm = MockLLMClient()
        supervisor = Supervisor(mock_llm, tool_reg, agent_reg, preprocessor=None)
        
        # Check results
        registered = agent_reg.list_agents()
        print(f"  Registered: {registered}")
        print(f"  Expected: {test['expected']}")
        
        if set(registered) == set(test['expected']):
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Got {registered}, expected {test['expected']}")
    
    # Restore config
    config["agents_available"] = original_available
    config["agents_unavailable"] = original_unavailable
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_config_filtering()
