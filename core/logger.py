"""
Simple interaction logger for agent workflows.
Captures conversations for later analysis and visualization.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import config


class InteractionLogger:
    """Logs agent interactions to JSON for later analysis."""
    
    def __init__(self, log_dir: str = None):
        """
        Args:
            log_dir: Directory for logs (defaults to ./logs)
        """
        self.log_dir = Path(log_dir) if log_dir else Path(config.get("log_dir"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = None
        self.start_time = None

    def _persist(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.current_log, f, indent=2)
    
    def start_session(
        self,
        task: str,
        config: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new logging session.
        
        Args:
            task: The user's task/query
            config: Configuration used (model, temperature, etc.)
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        self.start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = self.log_dir / f"{session_id}.json"

        # Redact any API keys from config
        redacted_config = {k: ("[REDACTED]" if "api_key" in k.lower() else v) for k, v in config.items()}
        
        self.current_log = {
            "session_id": session_id,
            "timestamp": timestamp,
            "task": task,
            "config": redacted_config,
            "interactions": [],
            "duration": None
        }
        
        return session_id
    
    def log_agent_start(
        self,
        agent_name: str,
        message: str,
        context: Optional[Dict] = None,
        parent_agent: Optional[str] = None
    ) -> int:
        """
        Log the start of an agent interaction.
        
        Returns:
            Interaction index for later updates
        """
        if not self.current_log:
            return -1
        
        interaction = {
            "agent": agent_name,
            "parent": parent_agent,
            "message": message,
            "context": context,
            "start_time": time.time() - self.start_time,
            "iterations": [],
            "result": None,
            "duration": None
        }
        
        self.current_log["interactions"].append(interaction)

        self._persist() 

        return len(self.current_log["interactions"]) - 1
    
    def log_iteration(
        self,
        interaction_idx: int,
        iteration: int,
        response_content: str,
        tool_calls: list,
        model_info: Dict[str, Any],
        tool_call_results: list = None,
        messages_sent: List[Dict[str, Any]] = None,
    ):
        """
        Log a single iteration within an agent interaction.
        
        Args:
            interaction_idx: Index of the interaction
            iteration: Iteration number
            response_content: LLM response content
            tool_calls: List of tool calls made
            model_info: Model metadata (usage, finish_reason, etc.)
            tool_call_results: Results from tool executions
            messages_sent: Optional full message history sent to LLM
        """
        if not self.current_log or interaction_idx < 0:
            return
        
        interaction = self.current_log["interactions"][interaction_idx]
        
        iteration_data = {
            "iteration": iteration,
            "timestamp": time.time() - self.start_time,
            "response": response_content,
            "tool_calls": [
                {
                    "name": tc["name"],
                    "arguments": tc.get("arguments", {}),
                    "id": tc.get("id")
                }
                for tc in tool_calls
            ],
            "model_info": model_info,
            "tool_call_results": tool_call_results
        }
        
        # Optionally include full message history for complete traceability
        if config.get('log_full_messages', False) and messages_sent:
            iteration_data["messages_sent"] = messages_sent
        
        interaction["iterations"].append(iteration_data)

        self._persist() 

    def log_agent_end(
        self,
        interaction_idx: int,
        result: str,
        total_tool_calls: int
    ):
        """Log the end of an agent interaction."""
        if not self.current_log or interaction_idx < 0:
            return
        
        interaction = self.current_log["interactions"][interaction_idx]
        interaction["result"] = result
        interaction["duration"] = time.time() - self.start_time - interaction["start_time"]
        interaction["total_tool_calls"] = total_tool_calls

        self._persist() 
    
    def end_session(self, final_result: str) -> Path:
        """
        End the logging session and save to file.
        
        Args:
            final_result: The final result returned to user
            
        Returns:
            Path to the saved log file
        """
        if not self.current_log:
            return None
        
        self.current_log["duration"] = time.time() - self.start_time
        self.current_log["final_result"] = final_result
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.current_log, f, indent=2)
        
        return self.log_file
    
    def get_session_log(self) -> Optional[Dict]:
        """Get the current session log."""
        return self.current_log


# Singleton instance
_logger = None

def get_logger() -> InteractionLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = InteractionLogger()
    return _logger
