"""
Base workflow abstraction.

All workflows inherit from this and implement their own orchestration logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Workflow(ABC):
    """
    Abstract base class for workflows that orchestrate the Supervisor.
    
    A workflow defines HOW the supervisor executes tasks (single run, retry loop,
    multi-stage pipeline, etc.) while the supervisor defines WHAT gets done
    (tool creation, agent delegation, code execution).
    """
    
    def __init__(self, supervisor):
        """
        Initialize workflow with a supervisor instance.
        
        Args:
            supervisor: Supervisor instance to orchestrate
        """
        self.supervisor = supervisor
    
    @abstractmethod
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the workflow on a task.
        
        Args:
            task: User's task description
            context: Optional context dict to pass to supervisor
            
        Returns:
            Dict with:
                - content: Final result text
                - tool_calls: List of all tool calls made
                - history: Agent interaction history
                - [workflow-specific fields]
        """
        pass
    
    def get_name(self) -> str:
        """Get workflow name for logging."""
        return self.__class__.__name__
