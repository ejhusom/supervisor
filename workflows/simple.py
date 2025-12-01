"""
Simple workflow - direct passthrough to supervisor.

This is the default workflow and matches the original system behavior.
"""

from typing import Dict, Any, Optional
from .base import Workflow


class SimpleWorkflow(Workflow):
    """
    Simple workflow that directly calls supervisor.run() once.
    
    This is the default workflow and preserves the original system behavior.
    Use this when you don't need evaluation, retry logic, or multi-stage execution.
    """
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute task by calling supervisor once.
        
        Args:
            task: User's task
            context: Optional context
            
        Returns:
            Supervisor's result dict
        """
        return self.supervisor.run(task, context)
