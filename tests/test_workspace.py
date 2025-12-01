#!/usr/bin/env python3
"""
Test script for unified workspace and code execution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from core.config import config
from core.sandbox import Sandbox
from core.standard_tools import get_standard_tools


def test_workspace_structure():
    """Test that workspace directories are created."""
    print("Testing workspace structure...")
    
    workspace = Path(config["workspace"])
    workspace_data = Path(config["workspace_data"])
    workspace_tools = Path(config["workspace_tools"])
    workspace_agents = Path(config["workspace_agents"])
    
    assert workspace.exists(), "Workspace root should exist"
    assert workspace_data.exists(), "Workspace data should exist"
    assert workspace_tools.exists(), "Workspace tools should exist"
    assert workspace_agents.exists(), "Workspace agents should exist"
    
    print(f"✓ Workspace structure created:")
    print(f"  - Root: {workspace}")
    print(f"  - Data: {workspace_data}")
    print(f"  - Tools: {workspace_tools}")
    print(f"  - Agents: {workspace_agents}")
    print()


def test_sandbox():
    """Test sandbox code execution."""
    print("Testing sandbox...")
    
    sandbox = Sandbox()
    
    # Test Python execution
    result = sandbox.execute("""
result = 2 + 2
print(f"Calculation: {result}")
""")
    
    assert result["success"], "Sandbox execution should succeed"
    assert result["return_value"] == 4, "Result should be 4"
    assert "Calculation: 4" in result["output"], "Output should contain calculation"
    
    print(f"✓ Sandbox execution works")
    print(f"  - Working directory: {sandbox.workspace}")
    print(f"  - Result: {result['return_value']}")
    print()


def test_standard_tools():
    """Test standard tools."""
    print("Testing standard tools...")
    
    sandbox = Sandbox()
    tools = get_standard_tools(sandbox)
    
    # Check all expected tools exist
    expected = [
        "execute_python", "run_command", "run_shell",
        "read_file", "write_file", "list_files", "pwd"
    ]
    
    for tool_name in expected:
        assert tool_name in tools, f"Tool {tool_name} should exist"
    
    print(f"✓ Standard tools available: {', '.join(expected)}")
    
    # Test pwd
    pwd_result = tools["pwd"]["function"]()
    print(f"  - pwd: {pwd_result}")
    
    # Test write and read
    tools["write_file"]["function"]("test.txt", "Hello, world!")
    content = tools["read_file"]["function"]("test.txt")
    assert "Hello, world!" in content, "File content should match"
    print(f"  - write_file/read_file: working")
    
    # Test list_files
    files = tools["list_files"]["function"]()
    assert any("test.txt" in f for f in files), "test.txt should be listed"
    print(f"  - list_files: {len(files)} items")
    
    # Test execute_python
    exec_result = tools["execute_python"]["function"]("result = 10 * 5")
    assert exec_result["success"], "Python execution should succeed"
    assert exec_result["return_value"] == 50, "Result should be 50"
    print(f"  - execute_python: working")
    
    print()


def test_integration():
    """Test that everything works together."""
    print("Testing integration...")
    
    from core.llm_client import LLMClient
    from registry.tool_registry import ToolRegistry
    from registry.agent_registry import AgentRegistry
    
    # These should use the unified workspace
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry()
    
    print(f"✓ Tool registry initialized")
    print(f"  - Persist dir: {tool_registry.persist_dir}")
    print(f"✓ Agent registry initialized")
    print(f"  - Persist dir: {agent_registry.persist_dir}")
    print()


def main():
    print("=" * 70)
    print("Testing Unified Workspace System")
    print("=" * 70)
    print()
    
    try:
        test_workspace_structure()
        test_sandbox()
        test_standard_tools()
        test_integration()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        print()
        print("The unified workspace system is working correctly.")
        print(f"Workspace location: {config['workspace']}")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
