"""
Standard agents: reusable, preconfigured Agent instances.

This module provides a lightweight skeleton for defining and registering
"standard" agents that the Supervisor (or other workflows) can load
and persist via the AgentRegistry.

Usage pattern (example):

	from core.llm_client import LLMClient
	from core.standard_tools import get_standard_tools
	from registry.agent_registry import AgentRegistry
	from core.standard_agents import register_standard_agents

	llm = LLMClient()
	tools = get_standard_tools(sandbox)
	registry = AgentRegistry()
	registered = register_standard_agents(registry, llm, tools)
	# registered -> list of agent names that were persisted

Notes:
- Agents here should use only tools available in `available_tools`.
- Preprocessing-dependent agents should gracefully skip if tools are missing.
"""

from typing import Dict, Any, List

from .agent import Agent


# Minimal, discoverable definitions for standard agents.
# Extend this list to add more agents or adjust prompts/tool sets.
STANDARD_AGENT_SPECS: List[Dict[str, Any]] = [
	# {
	# 	"name": "file_manager",
	# 	"system_prompt": (
	# 		"You are a precise file manager. You read and write files inside the "
	# 		"workspace, list directories, and never access paths outside the sandbox."
	# 	),
	# 	"tools": ["read_file", "write_file", "list_files"],
	# },
	# {
	# 	# Will only be created if preprocessing tools exist
	# 	"name": "log_analyst",
	# 	"system_prompt": (
	# 		"You analyze large logs using SQL queries and, if available, embeddings. "
	# 		"Prefer database queries over raw file scanning."
	# 	),
	# 	"tools": [
	# 		"query_logs_from_sqlite_database",
	# 		"get_error_logs_from_sqlite_database",
	# 		"search_similar_from_embeddings",
	# 	],
	# },
]


def _resolve_tools_for_spec(
	spec: Dict[str, Any], available_tools: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
	"""Return a filtered tool dict for a spec based on availability."""
	resolved = {}
	for tool_name in spec.get("tools", []):
		if tool_name in available_tools:
			resolved[tool_name] = available_tools[tool_name]
	return resolved


def build_standard_agents(
	llm_client: Any, available_tools: Dict[str, Dict[str, Any]]
) -> Dict[str, Agent]:
	"""Instantiate standard agents using the provided LLM client and tools.

	Agents whose tool sets resolve to empty (e.g., preprocessing not enabled)
	are skipped.
	"""
	agents: Dict[str, Agent] = {}

	for spec in STANDARD_AGENT_SPECS:
		tools_for_agent = _resolve_tools_for_spec(spec, available_tools)
		if not tools_for_agent:
			# Skip agents with no usable tools in current runtime
			continue

		agent = Agent(
			name=spec["name"],
			system_prompt=spec["system_prompt"],
			llm_client=llm_client,
			tools=tools_for_agent,
		)
		agents[spec["name"]] = agent

	return agents


def _persist_config_for_agent(spec: Dict[str, Any], tools_for_agent: Dict[str, Any]) -> Dict[str, Any]:
	"""Create a minimal persistence config for AgentRegistry."""
	return {
		"system_prompt": spec.get("system_prompt", ""),
		"tools": list(tools_for_agent.keys()),
	}


def register_standard_agents(
	agent_registry: Any,
	llm_client: Any,
	available_tools: Dict[str, Dict[str, Any]],
) -> List[str]:
	"""Build and register standard agents in the provided registry.

	Returns a list of agent names that were successfully registered and persisted.
	"""
	registered: List[str] = []
	for spec in STANDARD_AGENT_SPECS:
		tools_for_agent = _resolve_tools_for_spec(spec, available_tools)
		if not tools_for_agent:
			continue

		agent = Agent(
			name=spec["name"],
			system_prompt=spec["system_prompt"],
			llm_client=llm_client,
			tools=tools_for_agent,
		)

		config = _persist_config_for_agent(spec, tools_for_agent)
		agent_registry.register(spec["name"], agent, config)
		registered.append(spec["name"])

	return registered


__all__ = [
	"STANDARD_AGENT_SPECS",
	"build_standard_agents",
	"register_standard_agents",
]

