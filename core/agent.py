"""
Minimal agent implementation.
Just wraps LLM with tool execution - no complex workflows.
"""

from typing import List, Dict, Any, Optional
import json

from .config import config
from .logger import get_logger
from .printer import PrintManager

class Agent:
    """Simple agent: LLM + tools."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: Any,
        provider: str = None,
        model: str = None,
        tools: Dict[str, Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
        logging_enabled: bool = None,
    ):
        """
        Args:
            name: Agent identifier
            system_prompt: System instructions
            llm_client: LLMClient instance
            tools: Dict of {name: {"function": callable, "schema": dict}}
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.tools = tools or {}
        self.history = []
        self.provider = provider if provider is not None else config.get("provider", None)
        self.model = model if model is not None else config.get("model", None)
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.0)
        self.max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 8192)
        self.logging_enabled = logging_enabled if logging_enabled is not None else config.get("logging_enabled")

        self.logger = get_logger() if self.logging_enabled else None
        self.current_interaction_idx = None
        
        # Initialize printer
        self.printer = PrintManager(
            verbosity=config.get("print_verbosity", "normal"),
            use_colors=config.get("print_use_colors", True),
            truncate_length=config.get("print_truncate_length", 200)
        )
    
    def run(
        self,
        message: str,
        context: Dict[str, Any] = None,
        max_iterations: int = 20,
        parent_agent: str = None,
    ) -> Dict[str, Any]:
        """
        Execute agent on a message.
        
        Returns:
            {
                "content": str,
                "tool_calls": List[Dict],
                "history": List[Dict]
            }
        """

        if self.logger:
            self.current_interaction_idx = self.logger.log_agent_start(
                agent_name=self.name,
                message=message,
                context=context,
                parent_agent=parent_agent
            )

        # Print agent start
        self.printer.agent_start(self.name, parent_agent)

        messages = []
        
        # Add context if provided
        if context:
            context_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_text}\n\nTask: {message}"
            })
        else:
            messages.append({"role": "user", "content": message})
        
        # Get tool schemas
        tool_schemas = self._get_tool_schemas() if self.tools else None
        
        all_tool_calls = []
        iteration = 0
        
        # Agentic loop
        while iteration < max_iterations:
            iteration += 1
            
            # Print iteration header
            self.printer.iteration_header(iteration, max_iterations, self.name)
            
            # Call LLM
            response = self.llm_client.complete(
                messages=messages,
                system=self.system_prompt,
                tools=tool_schemas,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            self.history.append({
                "iteration": iteration,
                "response": response
            })
            
            # No tool calls? Done
            if not response["tool_calls"]:
                # Show response preview
                self.printer.response_preview(response["content"])
                self.printer.agent_end()
                
                # Log iteration without tool calls
                if self.logger and self.current_interaction_idx is not None:
                    self.logger.log_iteration(
                        interaction_idx=self.current_interaction_idx,
                        iteration=iteration,
                        response_content=response["content"],
                        tool_calls=response["tool_calls"],
                        model_info={
                            "model": response.get("model"),
                            "usage": response.get("usage"),
                            "finish_reason": response.get("finish_reason")
                        },
                        messages_sent=messages
                    )

                # End logging of agent
                if self.logger and self.current_interaction_idx is not None:
                    self.logger.log_agent_end(
                        interaction_idx=self.current_interaction_idx,
                        result=response["content"],
                        total_tool_calls=len(all_tool_calls)
                    )

                return {
                    "content": response["content"],
                    "tool_calls": all_tool_calls,
                    "history": self.history
                }
            
            # Execute tools
            messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": self._format_tool_calls(response["tool_calls"])
            })
            
            tool_results = []
            
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                all_tool_calls.append(tool_call)
                
                # Print tool call
                self.printer.tool_call(tool_name, tool_call["arguments"])
                
                # Execute tool
                try:
                    result = self._execute_tool(tool_name, tool_call["arguments"])
                    result_content = json.dumps(result) if isinstance(result, dict) else str(result)
                except Exception as e:
                    result_content = f"Error: {str(e)}"

                # Limit output length
                max_length = config.get("tool_call_output_max_length")

                if isinstance(result_content, str) and max_length > 0 and len(result_content) > max_length:
                    if config.get("truncate_tool_calls", True):
                        result_content = result_content[:max_length] + f"...[truncated]. This content was truncated at {max_length} characters. Full output length: {len(result_content)} characters."
                    else:
                        result_content = "Output too long to display."
                
                # Print tool result
                self.printer.tool_result(tool_name, result_content)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_name,
                    "content": result_content
                })
                
                tool_results.append({
                    "name": tool_name,
                    "arguments": tool_call["arguments"],
                    "id": tool_call.get("id"),
                    "result": result_content,
                })

            if self.logger and self.current_interaction_idx is not None:
                self.logger.log_iteration(
                    interaction_idx=self.current_interaction_idx,
                    iteration=iteration,
                    response_content=response["content"],
                    tool_calls=response["tool_calls"],
                    model_info={
                        "model": response.get("model"),
                        "usage": response.get("usage"),
                        "finish_reason": response.get("finish_reason")
                    },
                    tool_call_results=tool_results,
                    messages_sent=messages
                )

        # Max iterations reached
        self.printer.agent_end()
        
        if self.logger and self.current_interaction_idx is not None:
            self.logger.log_agent_end(
                interaction_idx=self.current_interaction_idx,
                result="Max iterations reached",
                total_tool_calls=len(all_tool_calls)
            )

        return {
            "content": "Max iterations reached",
            "tool_calls": all_tool_calls,
            "history": self.history
        }
    
    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM."""
        return [tool["schema"] for tool in self.tools.values()]
    
    def _execute_tool(self, name: str, arguments: str) -> Any:
        """Execute a tool."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        return self.tools[name]["function"](**args)
    
    def _format_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Format tool calls for LLM API."""
        return [{
            "id": tc.get("id"),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"] if isinstance(tc["arguments"], str) 
                            else json.dumps(tc["arguments"])
            }
        } for tc in tool_calls]
