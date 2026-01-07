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
	{
		# Will only be created if preprocessing tools exist
		"name": "log_analyst",
		"system_prompt": (
			"You analyze large logs using SQL queries and, if available, embeddings. "
			"Prefer database queries over raw file scanning."
		),
		"tools": [
			"query_logs_from_sqlite_database",
			"get_error_logs_from_sqlite_database",
			"search_similar_from_embeddings",
		],
	},
	{
		"name": "log_parser",
		"system_prompt": (
			"You extract log templates from raw log messages. "
			"A log message consists of: (1) Template - constant strings describing system events, "
			"and (2) Variables - dynamic runtime values. "
			"Your task: Replace all variable parts (IPs, IDs, numbers, paths) with <*> placeholders. "
			"Output ONLY the template - no explanations, no extra text. "
			"Focus on the message body, exclude log headers (timestamp, level, class). "
			"\n\nExamples:\n"
			"Log: 'Receiving block blk_1724757848743533110 src: /10.251.111.130:49851 dest: /10.251.111.130:50010'\n"
			"Template: 'Receiving block <*> src: <*>:<*> dest: <*>:<*>'\n"
			"\n"
			"Log: 'PacketResponder 1 for block blk_38865049064139660 terminating'\n"
			"Template: 'PacketResponder <*> for block <*> terminating'"
		),
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_template_critic",
		"system_prompt": (
			"You validate log templates. Given an original log message and a proposed template, "
			"verify if all variables are correctly abstracted with <*> placeholders. "
			"Rules:\n"
			"1. Focus on message body only (ignore headers)\n"
			"2. All dynamic values (IPs, ports, IDs, numbers, paths) must be <*>\n"
			"3. Preserve all constant text and punctuation exactly\n"
			"4. If correct: return template unchanged\n"
			"5. If incorrect: return corrected template only\n"
			"Output format: Single line, template only, no explanations."
		),
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_template_refiner",
		"system_prompt": (
			"You refine log templates by comparing two proposed templates (from parser and critic). "
			"Given: ORIGINAL_LOG, PARSER_TEMPLATE, CRITIC_TEMPLATE. "
			"Your task:\n"
			"1. Compare both templates for accuracy\n"
			"2. Produce the most accurate version (may be one of them, merged, or new)\n"
			"3. When unsure, prefer CRITIC_TEMPLATE\n"
			"4. All variables must be <*>, constant text preserved\n"
			"5. If both wrong, regenerate from ORIGINAL_LOG\n"
			"Output: Single line with final refined template only."
		),
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_anomaly_detector",
		"system_prompt": (
			"You detect anomalies in log sessions. Given a sequence of log messages from a session, "
			"determine if it represents normal (0) or anomalous (1) behavior. "
			"Detection criteria:\n"
			"- Textual anomalies: explicit errors/failures ('error', 'fail', 'exception', 'crash')\n"
			"- Behavioral anomalies: unusual sequences, missing events, irregular flow, abrupt termination\n"
			"Instructions:\n"
			"1. Parse each log, extract message body (ignore headers)\n"
			"2. Analyze sequence for textual and behavioral anomalies\n"
			"3. If ANY anomaly found: output 1\n"
			"4. If all normal: output 0\n"
			"Output format: Single digit (0 or 1), no explanation."
		),
		"tools": ["read_file", "query_logs_from_sqlite_database"],
	},
	{
		"name": "log_preprocessor",
		"system_prompt": (
			"You prepare log sessions for analysis. Extract message bodies from raw logs, "
			"removing headers (timestamp, level, class). "
			"Preserve event sequence and structure. "
			"Output clean, structured log data ready for anomaly detection."
		),
		"tools": ["read_file", "write_file", "run_command"],
	},
	{
		"name": "anomaly_critic",
		"system_prompt": (
			"You validate anomaly detection decisions. Given original logs and a detection result (0/1), "
			"verify correctness. "
			"Check for:\n"
			"1. Missed errors or exceptions\n"
			"2. False positives from normal operational messages\n"
			"3. Behavioral pattern inconsistencies\n"
			"Output: 0 (normal) or 1 (anomalous) - corrected decision if needed."
		),
		"tools": ["read_file"],
	},
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

