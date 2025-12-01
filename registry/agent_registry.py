"""
Dynamic agent registry.
Agents can be created at runtime by the supervisor.
"""

import json
from typing import Dict, Any
from pathlib import Path
from core.config import config


class AgentRegistry:
    """Manages dynamically created agents."""
    
    def __init__(self, persist_dir: str = None):
        """
        Args:
            persist_dir: Directory to save agent definitions (defaults to workspace/agents)
        """
        self.agents: Dict[str, Any] = {}  # Name -> Agent instance
        self.configs: Dict[str, Dict] = {}  # Name -> config for persistence
        self.persist_dir = Path(persist_dir) if persist_dir else Path(config.get("workspace_agents"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
    
    def register(
        self,
        name: str,
        agent: Any,
        config: Dict[str, Any] = None
    ) -> None:
        """
        Register an agent.
        
        Args:
            name: Agent name
            agent: Agent instance
            config: Config dict for persistence (system_prompt, tools)
        """
        self.agents[name] = agent
        
        if config:
            self.configs[name] = config
            self._persist_agent(name, config)
    
    def get(self, name: str) -> Any:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all registered agents."""
        return self.agents
    
    def list_agents(self) -> list:
        """List all agent names."""
        return list(self.agents.keys())
    
    def _persist_agent(self, name: str, config: Dict) -> None:
        """Save agent config to disk."""
        agent_file = self.persist_dir / f"{name}.json"
        
        with open(agent_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_agent_configs(self) -> Dict[str, Dict]:
        """Load all persisted agent configs."""
        configs = {}
        
        for agent_file in self.persist_dir.glob("*.json"):
            name = agent_file.stem
            
            with open(agent_file, 'r') as f:
                configs[name] = json.load(f)
        
        return configs
