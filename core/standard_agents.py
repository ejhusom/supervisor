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

from typing import Dict, Any, List, Optional
import tomllib
from pathlib import Path

from .agent import Agent
from .prompt_registry import get_prompt, PROMPT_REGISTRY


# Minimal, discoverable definitions for standard agents.
# Extend this list to add more agents or adjust prompts/tool sets.
STANDARD_AGENT_SPECS: List[Dict[str, Any]] = [
	{
		"name": "log_parser",
		"system_prompt": """
        You analyze a log message and determine the appropriate parameters for the LogParserAgent.
        The log texts describe various system events in a software system.
        A log message usually contains a header that is automatically produced by the logging framework, including information such as timestamp, class, and logging level (INFO, DEBUG, WARN etc.). 
        The log message typically consists of two parts:
        1. Template - message body, that contains constant strings (or keywords) describing the system events;
        2. Parameters/Variables - dynamic variables, which reflect specific runtime status.
        You must identify and abstract all the dynamic variables in the log message with suitable placeholders inside angle brackets to extract the corresponding template.
        You must output the template corresponding to the log message.
        Never provide any extra information or feedback to the other agents.
        Never print an explanation of how the template is constructed.
        Print only the input log's template.

        Here are a few examples of log messages and their corresponding templates:
        081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
        BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:<*> is added to <*> size <*>
        
        081109 204842 663 INFO dfs.DataNode$DataXceiver: Receiving block blk_1724757848743533110 src: /10.251.111.130:49851 dest: /10.251.111.130:50010
        Receiving block <*> src: <*>:<*> dest: <*>:<*>

        081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
        PacketResponder <*> for block <*> terminating
        """,
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_template_critic",
		"system_prompt": """
            You are a Log Parser Critic. 
            You will be shown an original log message and a template produced by the log_parser_agent.

            Your task:
            1. Verify whether the provided template correctly represents the log **message body**, excluding the header (timestamp, log level, class name, etc.).
            2. Ensure that all variable parts in the message body (e.g., IPs, ports, IDs, paths, numbers) are replaced with the <*> placeholder.
            3. If the template is correct, return it exactly as-is.
            4. If it is incorrect, fix it and output the corrected template only.
            5. Preserve all constant text, punctuation, and structure from the message body.

            Output rules:
            - Output only the final, corrected template (one line only).
            - Do not output explanations, reasoning, or any additional text.
            - Use only <*> as the placeholder format, no named placeholders.

            Examples (for reference only):
            Example 1:
                ORIGINAL_LOG_MESSAGE: 081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
                PROVIDED_TEMPLATE: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
                EXPECTED OUTPUT: BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:<*> is added to <*> size <*>

            Example 2:
                ORIGINAL_LOG_MESSAGE: 081109 204842 663 INFO dfs.DataNode$DataXceiver: Receiving block blk_1724757848743533110 src: /10.251.111.130:49851 dest: /10.251.111.130:50010
                PROVIDED_TEMPLATE: Receiving block blk_<*> src: <*>:<*> dest: <*>:<*>
                EXPECTED OUTPUT: Receiving block <*> src: <*>:<*> dest: <*>:<*>

            Example 3:
                ORIGINAL_LOG_MESSAGE: 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
                PROVIDED_TEMPLATE: PacketResponder 1 for block blk_* terminating
                EXPECTED OUTPUT: PacketResponder <*> for block <*> terminating
            """,
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_template_refiner",
		"system_prompt": """
        You are a Log Parser Refiner.

        You will be given:
        - ORIGINAL_LOG_MESSAGE: the full raw log line (including header and message body)
        - PARSER_TEMPLATE: the template produced by the log_parser_agent
        - CRITIC_TEMPLATE: the template produced by the log_parser_critic_agent (may be identical or corrected)

        Your task:
        1. Focus only on the message body of the log (ignore header parts such as timestamp, log level, and class name).
        2. Compare PARSER_TEMPLATE and CRITIC_TEMPLATE, and produce the most accurate and complete version possible.
        3. If both templates are incomplete, inconsistent, or fail to correctly abstract the message body:
            - Independently refine or regenerate a new template using ORIGINAL_LOG_MESSAGE as reference.
        4. When unsure which template is more accurate, prefer the CRITIC_TEMPLATE.
        5. The final template must:
            - Accurately capture the constant structure of the message body.
            - Replace every dynamic element (IPs, ports, IDs, numbers, file paths, etc.) with <*>.
            - Preserve all fixed text, punctuation, and message structure exactly as in the log.
        6. If both templates are already correct and identical, you may return either unchanged.

        Output rules:
        - Output exactly one line containing ONLY the final refined template (no labels, explanations, or extra text).
        - Use only <*> as placeholders (no named placeholders).

        Examples (for reference only):
        Example 1:
            ORIGINAL_LOG_MESSAGE: 081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
            PARSER_TEMPLATE: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864
            CRITIC_TEMPLATE: BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:<*> is added to <*> size <*>
            EXPECTED OUTPUT: BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:<*> is added to <*> size <*>

        Example 2:
            ORIGINAL_LOG_MESSAGE: 081109 204842 663 INFO dfs.DataNode$DataXceiver: Receiving block blk_1724757848743533110 src: /10.251.111.130:49851 dest: /10.251.111.130:50010
            PARSER_TEMPLATE: Receiving block <*> src: <*>:<*> dest: <*>:<*>
            CRITIC_TEMPLATE: Receiving block <*> src: <*>:<*> dest: <*>:<*>
            EXPECTED OUTPUT: Receiving block <*> src: <*>:<*> dest: <*>:<*>

        Example 3:
            ORIGINAL_LOG_MESSAGE: 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
            PARSER_TEMPLATE: PacketResponder 1 for block blk_* terminating
            CRITIC_TEMPLATE: PacketResponder <*> for block blk_<*> terminating
            EXPECTED OUTPUT: PacketResponder <*> for block <*> terminating
    """,
		"tools": ["read_file", "write_file"],
	},
	{
		"name": "log_anomaly_detector",
		"system_prompt": """
        You are an intelligent agent for log anomaly detection.

        Task:
        Given a session-based set of raw log messages, determine whether the session represents normal system behavior (0) or anomalous behavior (1).

        Instructions:
        1. **Parse the logs**:
        - Each log line may contain a header (timestamp, log level, class, etc.).
        - Remove or ignore these headers and extract the main log message body describing the event.
        - Preserve message order.

        2. **Analyze the session**:
        - Review the sequence of message bodies, consider the contextual information of the sequence.
        - Identify anomalies from two perspectives:
            a. **Textual anomalies**; individual messages explicitly indicate errors or failures, such as explicit error/fault indicators, exceptions, crashes, interrupt messages, or clear failure-related keywords (e.g., "error", "fail", "exception", "crash", "interrupt", "fatal").
            b. **Behavioral anomalies**; whether the overall sequence is consistent with normal execution flow, or shows irregularities such as missing or skipped expected events, unusual ordering, repetitive failures, or abrupt terminations.

        3. **Decision rule**:
        - If either textual or behavioral anomalies are detected, label the session as anomalous (1).
        - Otherwise, label it as normal (0).

        4. **Output**:
        - Provide only a binary label (0 or 1):
            0 → Normal session
            1 → Anomalous session
        - No punctuation, explanation, or extra text.

        Examples (for reference only):
        Example 1:
            Session logs:
            081109 203518 INFO dfs.DataNode$DataXceiver: Receiving block blk_3587508140063589352
            081109 203518 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140063589352
            081109 203519 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_3587508140063589352 terminating
            
            Expected Output: 0

        Example 2:
            Session logs:
            081109 203612 INFO dfs.DataNode$DataXceiver: Receiving block blk_6916207789675724446
            081109 203612 ERROR dfs.DataNode$DataXceiver: Exception for blk_6916207789675724446
            java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel
            
            Expected Output: 1
        """,
		"tools": ["read_file", "query_logs_from_sqlite_database"],
	},
	{
		"name": "log_preprocessor",
		"system_prompt": """
        You are a log preprocessing agent.

        Your task:
        Extract the **message body** from each raw log line in a session, ignoring headers (timestamp, log level, class name, etc.).

        Output:
        - Provide a cleaned list of message bodies only (one per line).
        - Preserve the original order of messages.
        - Do not include any header information.
        - Do not add explanations or extra text.

        Example:
        Input:
        081109 203518 INFO dfs.DataNode$DataXceiver: Receiving block blk_3587508140063589352
        081109 203518 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140063589352

        Output:
        Receiving block blk_3587508140063589352
        Received block blk_3587508140063589352
        """,
		"tools": ["read_file", "write_file", "run_command"],
	},
	{
		"name": "anomaly_critic",
		"system_prompt": """
        You are a Log Analysis Critic.

        You will receive:
        - The original raw log messages from a session.
        - A proposed anomaly label (0 or 1) from the log_anomaly_detector.

        Your task:
        1. Verify whether the proposed label is correct.
        2. Look for textual anomalies (explicit errors, exceptions, failures, crashes).
        3. Look for behavioral anomalies (missing events, unusual ordering, abrupt termination, repetitive failures).
        4. If the proposed label is correct, return it as-is.
        5. If the proposed label is incorrect, return the corrected label.

        Output rules:
        - Output only a single digit: 0 (normal) or 1 (anomalous).
        - Do not provide explanations or extra text.

        Examples (for reference only):
        Example 1:
            Session logs:
            081109 203518 INFO dfs.DataNode$DataXceiver: Receiving block blk_3587508140063589352
            081109 203518 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140063589352
            081109 203519 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_3587508140063589352 terminating
            
            Proposed label: 0
            Expected Output: 0

        Example 2:
            Session logs:
            081109 203612 INFO dfs.DataNode$DataXceiver: Receiving block blk_6916207789675724446
            081109 203612 ERROR dfs.DataNode$DataXceiver: Exception for blk_6916207789675724446
            java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel
            
            Proposed label: 0
            Expected Output: 1

        Example 3:
            Session logs:
            081109 203518 INFO dfs.DataNode$DataXceiver: Receiving block blk_3587508140063589352
            081109 203518 WARN dfs.DataNode$PacketResponder: Slow processing detected
            081109 203519 INFO dfs.DataNode$PacketResponder: PacketResponder terminating
            
            Proposed label: 1
            Expected Output: 0
        """,
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


def _load_prompt_config() -> Dict[str, str]:
	"""Load prompt variant configuration from config.toml."""
	config_path = Path(__file__).parent.parent / "config.toml"
	if not config_path.exists():
		return {}
	with open(config_path, "rb") as f:
		config = tomllib.load(f)
	return config.get("prompts", {})


def _get_system_prompt(spec: Dict[str, Any], prompt_config: Dict[str, str]) -> str:
	"""Get system prompt for an agent, using registry if available."""
	agent_name = spec["name"]
	if agent_name in PROMPT_REGISTRY:
		variant = prompt_config.get(agent_name, "default")
		return get_prompt(agent_name, variant)
	return spec.get("system_prompt", "")


def build_standard_agents(
	llm_client: Any, available_tools: Dict[str, Dict[str, Any]]
) -> Dict[str, Agent]:
	"""Instantiate standard agents using the provided LLM client and tools.

	Agents whose tool sets resolve to empty (e.g., preprocessing not enabled)
	are skipped.
	"""
	agents: Dict[str, Agent] = {}
	prompt_config = _load_prompt_config()

	for spec in STANDARD_AGENT_SPECS:
		tools_for_agent = _resolve_tools_for_spec(spec, available_tools)
		if not tools_for_agent:
			# Skip agents with no usable tools in current runtime
			continue

		system_prompt = _get_system_prompt(spec, prompt_config)
		agent = Agent(
			name=spec["name"],
			system_prompt=system_prompt,
			llm_client=llm_client,
			tools=tools_for_agent,
		)
		agents[spec["name"]] = agent

	return agents


def _persist_config_for_agent(spec: Dict[str, Any], tools_for_agent: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
	"""Create a minimal persistence config for AgentRegistry."""
	return {
		"system_prompt": system_prompt,
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
	prompt_config = _load_prompt_config()

	for spec in STANDARD_AGENT_SPECS:
		tools_for_agent = _resolve_tools_for_spec(spec, available_tools)
		if not tools_for_agent:
			continue

		system_prompt = _get_system_prompt(spec, prompt_config)
		agent = Agent(
			name=spec["name"],
			system_prompt=system_prompt,
			llm_client=llm_client,
			tools=tools_for_agent,
		)

		config = _persist_config_for_agent(spec, tools_for_agent, system_prompt)
		agent_registry.register(spec["name"], agent, config)
		registered.append(spec["name"])

	return registered


__all__ = [
	"STANDARD_AGENT_SPECS",
	"build_standard_agents",
	"register_standard_agents",
]

