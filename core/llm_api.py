import os
import requests

def completion(
    provider,
    model,
    messages,
    api_key=None,
    base_url=None,
    temperature=None,
    top_p=None,
    n=None,
    stream=None,
    stop=None,
    max_tokens=None,
    presence_penalty=None,
    frequency_penalty=None,
    logit_bias=None,
    user=None,
    functions=None,
    function_call=None,
    **kwargs
):
    """
    Create a completion for the provided messages and parameters.
    Compatible with OpenAI, Anthropic, and Ollama APIs.
    
    Args:
        provider: (required) LLM provider ("openai", "anthropic", "ollama")
        model: (required) Model ID to use
        messages: (required) List of message dicts with 'role' and 'content'
        api_key: API key (or from OPENAI_API_KEY/ANTHROPIC_API_KEY env vars)
        base_url: Override base URL
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        n: Number of completions to generate
        stream: Whether to stream responses
        stop: Stop sequences (string or list)
        max_tokens: Maximum tokens to generate
        presence_penalty: Presence penalty (-2 to 2)
        frequency_penalty: Frequency penalty (-2 to 2)
        logit_bias: Token logit biases
        user: Unique user identifier
        functions: List of functions for function calling
        function_call: Control function calling behavior
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Dictionary in LiteLLM format with choices, usage, etc.
    """
    # Build params dict from non-None values
    params = {k: v for k, v in {
        'temperature': temperature,
        'top_p': top_p,
        'n': n,
        'stream': stream,
        'stop': stop,
        'max_tokens': max_tokens,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'logit_bias': logit_bias,
        'user': user,
        'functions': functions,
        'function_call': function_call,
    }.items() if v is not None}
    params.update(kwargs)
    
    if provider == "openai":
        return _query_openai(messages, model, api_key, base_url, **params)
    elif provider == "anthropic":
        return _query_anthropic(messages, model, api_key, **params)
    elif provider == "ollama":
        return _query_ollama(messages, model, base_url, **params)
    else:
        raise ValueError(f"Unknown provider for model: {model}")

def _query_openai(messages, model, api_key, base_url, **kwargs):
    url = f"{base_url or 'https://api.openai.com'}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key or os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # OpenAI already uses this format
    return {
        'choices': result['choices'],
        'created': result.get('created'),
        'model': result.get('model'),
        'usage': result.get('usage', {})
    }

def _query_anthropic(messages, model, api_key, **kwargs):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key or os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    
    # Filter to Anthropic-supported parameters
    supported = ['max_tokens', 'temperature', 'top_p', 'top_k', 'stop_sequences', 'stream', 'system']
    anthropic_kwargs = {k: v for k, v in kwargs.items() if k in supported}
    
    # Map 'stop' to 'stop_sequences' for Anthropic
    if 'stop' in kwargs and 'stop_sequences' not in anthropic_kwargs:
        stop = kwargs['stop']
        anthropic_kwargs['stop_sequences'] = [stop] if isinstance(stop, str) else stop
    
    # Anthropic requires max_tokens
    if 'max_tokens' not in anthropic_kwargs:
        anthropic_kwargs['max_tokens'] = 1024
    
    data = {
        "model": model,
        "messages": messages,
        **anthropic_kwargs
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # Convert to LiteLLM format
    return {
        'choices': [{
            'finish_reason': result.get('stop_reason', 'stop'),
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': result['content'][0]['text']
            }
        }],
        'created': None,
        'model': result.get('model'),
        'usage': {
            'prompt_tokens': result.get('usage', {}).get('input_tokens', 0),
            'completion_tokens': result.get('usage', {}).get('output_tokens', 0),
            'total_tokens': result.get('usage', {}).get('input_tokens', 0) + result.get('usage', {}).get('output_tokens', 0)
        }
    }

def _query_ollama(messages, model, base_url, **kwargs):
    url = f"{base_url or 'http://localhost:11434'}/api/generate"
    
    # Convert messages to prompt
    prompt = _messages_to_prompt(messages)
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **kwargs
    }
    resp = requests.post(url, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # Convert to LiteLLM format
    return {
        'choices': [{
            'finish_reason': 'stop' if result.get('done') else 'length',
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': result['response']
            }
        }],
        'created': None,
        'model': result.get('model'),
        'usage': {
            'prompt_tokens': result.get('prompt_eval_count', 0),
            'completion_tokens': result.get('eval_count', 0),
            'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
        }
    }

def _messages_to_prompt(messages):
    """Convert OpenAI messages format to a simple prompt string."""
    prompt_parts = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            prompt_parts.append(f"System: {content}")
        elif role == 'user':
            prompt_parts.append(f"User: {content}")
        elif role == 'assistant':
            prompt_parts.append(f"Assistant: {content}")
    return '\n\n'.join(prompt_parts)

if __name__ == "__main__":
    # Example usage
    messages = [{"role": "user", "content": "What is 2+2?"}]
    
    # OpenAI
    # result = completion(model="gpt-4", messages=messages)
    
    # Anthropic
    # result = completion(model="claude-sonnet-4-20250514", messages=messages)
    
    # Ollama
    result = completion(model="llama2", messages=messages)
    
    # Extract response
    content = result['choices'][0]['message']['content']
    print(f"Response: {content}")
    print(f"Tokens used: {result['usage']['total_tokens']}")