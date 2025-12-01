"""
Simplified LLM client using LiteLLM.
"""
import os
from typing import List, Dict, Optional, Any
from litellm import completion
import litellm

class LLMClient:
    """Unified LLM interface."""
    
    def __init__(self, 
                 provider: str,
                 model: str,
                 api_key: Optional[str] = None,
    ):
        """Initialize LLM client."""
        self.provider = provider
        self.api_key = api_key
        self.model = model

        if self.provider in ["anthropic", "openai"] and not self.api_key:
            raise ValueError(f"API key required for provider: {self.provider}")

        if self.provider == "anthropic" and self.api_key:
            os.environ["ANTHROPIC_KEY"] = self.api_key
        elif self.provider == "openai" and self.api_key:
            os.environ["OPENAI_KEY"] = self.api_key

        # For Ollama, prefix model name with "ollama_chat/"
        if self.provider == "ollama" and not model.startswith("ollama/") and not model.startswith("ollama_chat/"):
            self.model = f"ollama_chat/{model}"

    def complete(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 16384,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Make completion request.
        
        Returns:
            {
                "content": str,
                "tool_calls": List[Dict],
                "usage": Dict
            }
        """

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "drop_params": True
        }
        
        # Add system prompt
        if system:
            kwargs["messages"] = [{"role": "system", "content": system}] + messages
        
        # Add tools
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = completion(**kwargs)
            return self._parse_response(response)
        
        except Exception as e:
            raise Exception(f"LLM completion failed: {str(e)}")
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LiteLLM response."""
        content = ""
        tool_calls = []
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            if hasattr(message, 'content') and message.content:
                content = message.content
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id if hasattr(tc, 'id') else None,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    })

            finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None
            role = message.role if hasattr(message, 'role') else None

        
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            usage = {
                "input_tokens": getattr(usage_obj, 'prompt_tokens', 0),
                "output_tokens": getattr(usage_obj, 'completion_tokens', 0),
                "total_tokens": getattr(usage_obj, 'total_tokens', 0)
            }

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
            "created": getattr(response, 'created', None),
            "model": getattr(response, 'model', None),
            "role": role,
            "finish_reason": finish_reason
        }
