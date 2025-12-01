#!/usr/bin/env python3
"""
Structured query wrapper for iExplain.

Non-invasive wrapper that requests JSON output from the supervisor
without modifying core code.
"""

import json
import re
from typing import Dict, Any, Optional


def query_with_structured_output(supervisor, question: str, schema: Dict = None) -> Any:
    """
    Query iExplain and enforce structured JSON output.
    
    Args:
        supervisor: Supervisor instance
        question: Question to ask
        schema: Optional JSON schema to include in prompt
    
    Returns:
        Parsed answer (int, str, list, or dict depending on schema)
    """
    # Build enhanced question that requests JSON
    if schema:
        schema_str = json.dumps(schema, indent=2)
        enhanced_question = f"""{question}

CRITICAL: You MUST respond with ONLY a valid JSON object. Do not include any explanatory text before or after the JSON.

Expected JSON schema:
{schema_str}

Respond with only the JSON object, nothing else."""
    else:
        enhanced_question = f"""{question}

CRITICAL: You MUST respond with ONLY a valid JSON value (number, string, list, or object). Do not include any explanatory text before or after the JSON.

Examples:
- For a number: 42
- For a string: "KERNEL"
- For a list: ["KERNEL", "APP", "RAS"]

Respond with only the JSON value, nothing else."""
    
    # Run supervisor
    result = supervisor.run(enhanced_question)
    
    # Extract JSON from response
    return extract_json(result["content"])


def extract_json(text: str) -> Any:
    """
    Extract JSON from text that may contain markdown or explanatory text.
    
    Args:
        text: Text potentially containing JSON
    
    Returns:
        Parsed JSON value
    
    Raises:
        ValueError: If no valid JSON found
    """
    # Try to extract from markdown code blocks first
    if "```json" in text:
        match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    if "```" in text:
        match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON object/array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r'\{[^{}]*\}',  # Simple object
        r'\{.*?\}',     # Object with nesting
        r'\[.*?\]',     # Array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to parse entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract a simple value
    # Look for a single number
    number_match = re.search(r'\b(\d+)\b', text)
    if number_match:
        return int(number_match.group(1))
    
    # Look for a quoted string
    string_match = re.search(r'"([^"]+)"', text)
    if string_match:
        return string_match.group(1)
    
    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


def create_answer_schema(question_spec: Dict) -> Optional[Dict]:
    """
    Create JSON schema from question specification.
    
    Args:
        question_spec: Question dict from ground truth
    
    Returns:
        JSON schema dict or None
    """
    answer_type = question_spec.get("answer_type")
    
    if answer_type == "integer":
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "integer"}
            },
            "required": ["answer"]
        }
    
    elif answer_type == "string_match":
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }
    
    elif answer_type == "list":
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["answer"]
        }
    
    return None


def query_question(supervisor, question_spec: Dict) -> Any:
    """
    Query a specific evaluation question.
    
    Args:
        supervisor: Supervisor instance
        question_spec: Question dict from ground truth
    
    Returns:
        Extracted answer value
    """
    schema = create_answer_schema(question_spec)
    result = query_with_structured_output(
        supervisor, 
        question_spec["question"],
        schema
    )
    
    # If result is a dict with "answer" key, extract it
    if isinstance(result, dict) and "answer" in result:
        return result["answer"]
    
    return result
