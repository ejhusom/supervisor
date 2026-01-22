"""
Prompt registry for swappable system prompts.
"""
from typing import Dict, Optional

# Import prompts from examples
from core.system_prompts import (
    SYS_MSG_LOG_PREPROCESSOR_FEW_SHOT,
    SYS_MSG_LOG_PREPROCESSOR_ZERO_SHOT,
    SYS_MSG_LOG_ANOMALY_DETECTOR_HDFS_FEW_SHOT,
    SYS_MSG_LOG_ANOMALY_DETECTOR_HDFS_ZERO_SHOT,
    SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_FEW_SHOT,
    SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_ZERO_SHOT,
    SYS_MSG_LOG_EXPLANATION_GENERATOR_FEW_SHOT,
    SYS_MSG_LOG_EXPLANATION_GENERATOR_ZERO_SHOT,
)

PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "log_preprocessor": {
        "few_shot": SYS_MSG_LOG_PREPROCESSOR_FEW_SHOT,
        "zero_shot": SYS_MSG_LOG_PREPROCESSOR_ZERO_SHOT,
        "default": SYS_MSG_LOG_PREPROCESSOR_FEW_SHOT,
    },
    "log_anomaly_detector": {
        "hdfs_few_shot": SYS_MSG_LOG_ANOMALY_DETECTOR_HDFS_FEW_SHOT,
        "hdfs_zero_shot": SYS_MSG_LOG_ANOMALY_DETECTOR_HDFS_ZERO_SHOT,
        "simplified_few_shot": SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_FEW_SHOT,
        "simplified_zero_shot": SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_ZERO_SHOT,
        "default": SYS_MSG_LOG_ANOMALY_DETECTOR_HDFS_FEW_SHOT,
    },
    "log_explainer": {
        "few_shot": SYS_MSG_LOG_EXPLANATION_GENERATOR_FEW_SHOT,
        "zero_shot": SYS_MSG_LOG_EXPLANATION_GENERATOR_ZERO_SHOT,
        "default": SYS_MSG_LOG_EXPLANATION_GENERATOR_FEW_SHOT,
    },
}


def get_prompt(agent_type: str, variant: str = "default") -> str:
    """Get a system prompt by agent type and variant."""
    if agent_type not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    prompts = PROMPT_REGISTRY[agent_type]
    if variant not in prompts:
        variant = "default"
    
    return prompts[variant]


def list_variants(agent_type: str) -> list:
    """List available prompt variants for an agent type."""
    if agent_type not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return [k for k in PROMPT_REGISTRY[agent_type].keys() if k != "default"]


def list_agent_types() -> list:
    """List all agent types with configurable prompts."""
    return list(PROMPT_REGISTRY.keys())
