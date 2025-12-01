"""
Supervisor: Main orchestration loop with meta-tools and preprocessing.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .agent import Agent
from .config import config
from .llm_client import LLMClient
from .logger import get_logger
from .sandbox import Sandbox
from .standard_tools import get_standard_tools
from .preprocessor import Preprocessor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

class Supervisor:
    """
    Supervisor agent with meta-tools for self-modification and preprocessing support.
    
    Can create tools, create agents, execute code, delegate tasks, and access preprocessed data.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        instructions_dir: str = "instructions",
        preprocessor: Optional[Preprocessor] = None
    ):
        """
        Args:
            llm_client: LLM client for API calls
            tool_registry: Dynamic tool registry
            agent_registry: Dynamic agent registry
            instructions_dir: Directory with markdown instructions
            preprocessor: Optional preprocessor with data access tools
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.sandbox = Sandbox()
        self.instructions_dir = Path(instructions_dir)
        self.preprocessor = preprocessor
        
        # Get standard tools
        self.standard_tools = get_standard_tools(self.sandbox)
        
        # Add preprocessing tools if available
        if preprocessor:
            preprocessing_tools = preprocessor.get_all_tools()
            self.standard_tools.update(preprocessing_tools)
            print(f"Added {len(preprocessing_tools)} preprocessing tools to supervisor")
        
        # Create supervisor agent with all tools
        all_tools = {**self._get_meta_tools(), **self.standard_tools}

        # If config.tools_available is non-empty, filter tools
        tools_available = config.get("tools_available", [])
        tools_unavailable = config.get("tools_unavailable", [])
        if tools_available:
            all_tools = {name: tool for name, tool in all_tools.items() if name in tools_available}
        if tools_unavailable:
            all_tools = {name: tool for name, tool in all_tools.items() if name not in tools_unavailable}

        system_prompt = self._load_system_prompt()
        # Add tool list with descriptions
        full_system_prompt = f"{system_prompt}\n\nYou have access to the following tools:\n"
        for tool_name, tool in all_tools.items():
            full_system_prompt += f"- {tool_name}: {tool["schema"]["function"]["description"]}\n"
        
        self.agent = Agent(
            name="supervisor",
            system_prompt=full_system_prompt,
            llm_client=llm_client,
            tools=all_tools,
        )
    
    def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute supervisor on a task."""
        # Start logging
        logger = get_logger()
        
        # Add preprocessing info to context if available
        # TODO: Consider whether this is needed.
        # if self.preprocessor and self.preprocessor.metadata:
        #     if context is None:
        #         context = {}
        #     context["preprocessing"] = self.preprocessor.metadata
        
        session_id = logger.start_session(
            task=message,
            config={
                **config,
                "system_prompt": self.agent.system_prompt,
                "tools": list(self.agent.tools.keys()),
                "preprocessing_enabled": self.preprocessor is not None
            }
        )

        result = self.agent.run(message, context)

        # End logging
        log_file = logger.end_session(final_result=result["content"])
        print(f"\nSession log saved to: {log_file}")

        return result
    
    def _load_system_prompt(self) -> str:
        """Load supervisor system prompt from instructions."""
        prompt_file = self.instructions_dir / "supervisor.md"
        
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                base_prompt = f.read()
        else:
            base_prompt = """You are a supervisor agent that orchestrates complex tasks.
-
-You can:
-- Create new tools by writing Python code
-- Create specialized agents for subtasks
-- Execute code and Unix commands
-- Read/write files
-- Delegate tasks to created agents
-
-Standard tools available: execute_python, run_command, run_shell, read_file, write_file, list_files, pwd
-
-Think step by step:
-1. Understand the task
-2. Identify what capabilities are needed
-3. Create tools/agents as needed
-4. Execute the task
-5. Return results
-
-Be strategic and efficient."""
        
        # Add preprocessing info if available
        if self.preprocessor:
            preprocessing_info = "\n\n## Preprocessing Tools Available\n\n"
            preprocessing_info += "The following preprocessing has been done on the input data:\n\n"
            
            for step_name, metadata in self.preprocessor.metadata.items():
                preprocessing_info += f"**{step_name}:**\n"
                for key, value in metadata.items():
                    preprocessing_info += f"- {key}: {value}\n"
                preprocessing_info += "\n"
            
            preprocessing_info += "You have access to specialized tools for querying this preprocessed data. "
            preprocessing_info += "Use these tools instead of trying to read large files directly.\n"
            
            return base_prompt + preprocessing_info
        
        return base_prompt
    
    def _get_meta_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get meta-tools for supervisor (tools that modify the system)."""
        return {
            "create_tool": {
                "function": self._create_tool,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "create_tool",
                        "description": "Create a new tool by providing Python code. The tool will be tested and registered if valid.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Tool name (lowercase, underscores)"
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Python function code. Must be a complete function definition."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What the tool does"
                                },
                                "parameters_schema": {
                                    "type": "object",
                                    "description": "JSON schema for function parameters"
                                }
                            },
                            "required": ["name", "code", "description", "parameters_schema"]
                        }
                    }
                }
            },
            "create_agent": {
                "function": self._create_agent,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "create_agent",
                        "description": "Create a specialized agent with specific tools and instructions.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Agent name"
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Agent's system instructions"
                                },
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of tool names (includes preprocessing tools if available)"
                                }
                            },
                            "required": ["name", "system_prompt", "tools"]
                        }
                    }
                }
            },
            "read_instructions": {
                "function": self._read_instructions,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "read_instructions",
                        "description": "Read a markdown instruction file for guidance.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Filename (e.g., 'tool_creation.md')"
                                }
                            },
                            "required": ["filename"]
                        }
                    }
                }
            },
            "delegate_to_agent": {
                "function": self._delegate_to_agent,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "delegate_to_agent",
                        "description": "Delegate a task to a created agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent_name": {
                                    "type": "string",
                                    "description": "Name of the agent to delegate to"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Task description for the agent"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Optional context to provide"
                                }
                            },
                            "required": ["agent_name", "task"]
                        }
                    }
                }
            },
            "list_tools": {
                "function": self._list_tools,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_tools",
                        "description": "List all available tools (registry, standard, and preprocessing tools).",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            },
            "list_agents": {
                "function": self._list_agents,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_agents",
                        "description": "List all created agents.",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            }
        }
    
    def _create_tool(
        self,
        name: str,
        code: str,
        description: str,
        parameters_schema: Dict
    ) -> Dict[str, Any]:
        """Create and register a new tool."""
        # Test the code first
        test_result = self.sandbox.execute(code)
        
        if not test_result["success"]:
            return {
                "success": False,
                "error": f"Code execution failed: {test_result['error']}"
            }
        
        # Extract function from code
        try:
            namespace = {}
            exec(code, namespace)
            
            # Find the function
            func = None
            for obj in namespace.values():
                if callable(obj) and hasattr(obj, '__name__') and obj.__name__ != '__builtins__':
                    func = obj
                    break
            
            if not func:
                return {
                    "success": False,
                    "error": "No function found in code"
                }
            
            # Create schema
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters_schema
                }
            }
            
            # Register tool
            self.tool_registry.register(name, func, schema, code)
            
            return {
                "success": True,
                "message": f"Tool '{name}' created and registered"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create tool: {str(e)}"
            }
    
    def _create_agent(
        self,
        name: str,
        system_prompt: str,
        tools: List[str]
    ) -> Dict[str, Any]:
        """Create and register a new agent."""
        # Collect tools from registry and standard tools
        agent_tools = {}
        
        for tool_name in tools:
            # Check standard tools first (includes preprocessing tools)
            if tool_name in self.standard_tools:
                agent_tools[tool_name] = self.standard_tools[tool_name]
            # Then check registry
            else:
                tool = self.tool_registry.get(tool_name)
                if tool:
                    agent_tools[tool_name] = tool
                else:
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found"
                    }
        
        # Create agent
        agent = Agent(
            name=name,
            system_prompt=system_prompt,
            llm_client=self.llm_client,
            tools=agent_tools
        )
        
        # Register agent
        config_data = {
            "system_prompt": system_prompt,
            "tools": tools
        }
        self.agent_registry.register(name, agent, config_data)
        
        return {
            "success": True,
            "message": f"Agent '{name}' created with {len(agent_tools)} tools"
        }
    
    def _read_instructions(self, filename: str) -> str:
        """Read instruction file."""
        file_path = self.instructions_dir / filename
        
        if not file_path.exists():
            return f"Error: Instruction file '{filename}' not found"
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def _delegate_to_agent(
        self,
        agent_name: str,
        task: str,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Delegate task to an agent."""
        agent = self.agent_registry.get(agent_name)
        
        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }
        
        result = agent.run(task, context, parent_agent="supervisor")
        
        return {
            "success": True,
            "agent": agent_name,
            "response": result["content"]
        }
    
    def _list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        registry_tools = self.tool_registry.list_tools()
        standard_tool_names = list(self.standard_tools.keys())
        
        # Separate preprocessing tools if available
        preprocessing_tool_names = []
        if self.preprocessor:
            preprocessing_tool_names = list(self.preprocessor.get_all_tools().keys())
        
        return {
            "registry_tools": registry_tools,
            "standard_tools": [t for t in standard_tool_names if t not in preprocessing_tool_names],
            "preprocessing_tools": preprocessing_tool_names,
            "all_tools": registry_tools + standard_tool_names
        }
    
    def _list_agents(self) -> Dict[str, list]:
        """List all created agents."""
        return {
            "agents": self.agent_registry.list_agents()
        }