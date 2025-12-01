#!/usr/bin/env python3
"""
Simple log viewer for agent interaction logs.
Displays the timeline and structure of agent interactions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_log(log_file: Path) -> Dict:
    """Load a log file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def print_timeline(log: Dict):
    """Print an enhanced ASCII timeline with visual bars."""
    total_duration = log.get('duration', 0)
    bar_width = 50  # Width of the visual timeline bar
    
    # Header
    print("═" * 80)
    try:
        print(f"Session: {log['session_id']} | Duration: {total_duration:.2f}s")
    except TypeError:
        print(f"Session: {log['session_id']} | Duration: N/A")
    print(f"Task: {log['task']}")
    print(f"Model: {log['config']['model']} (temp={log['config'].get('temperature', 0)})")
    print("═" * 80)
    print()
    
    # Helper to create visual bar
    def make_bar(start_time, duration, total):
        if not isinstance(duration, (int, float)) or not isinstance(total, (int, float)) or total == 0:
            return "░" * bar_width
        
        # Calculate position and width
        start_pos = int((start_time / total) * bar_width)
        bar_len = max(1, int((duration / total) * bar_width))
        
        # Build bar
        bar = ['░'] * bar_width
        for i in range(start_pos, min(start_pos + bar_len, bar_width)):
            bar[i] = '▓'
        return ''.join(bar)
    
    # Process each interaction
    for idx, interaction in enumerate(log['interactions']):
        agent_name = interaction['agent']
        duration = interaction.get('duration', 0)
        start_time = interaction.get('start_time', 0)
        parent = interaction.get('parent')
        
        # Agent header
        if parent:
            try:
                print(f"{agent_name} [{duration:.2f}s] (called by {parent})")
            except (TypeError, ValueError):
                print(f"{agent_name} [N/A s] (called by {parent})")
        else:
            try:
                print(f"{agent_name} [{duration:.2f}s]")
            except (TypeError, ValueError):
                print(f"{agent_name} [N/A s]")

        
        # Visual timeline bar
        bar = make_bar(start_time, duration, total_duration)
        print(f"{bar}")
        
        # Iterations
        iterations = interaction.get('iterations', [])
        for iter_idx, iter_data in enumerate(iterations):
            is_last = iter_idx == len(iterations) - 1
            prefix = "└─" if is_last else "├─"
            
            iter_num = iter_data['iteration']
            iter_time = iter_data['timestamp']
            tool_calls = iter_data.get('tool_calls', [])
            
            # Iteration header
            if tool_calls:
                tool_names = ', '.join(tc['name'] for tc in tool_calls)
                print(f"{prefix} Iter {iter_num} [{iter_time:.2f}s]: {tool_names}")
            else:
                print(f"{prefix} Iter {iter_num} [{iter_time:.2f}s]: (response)")
        
        print()
    
    # Footer with summary
    total_iterations = sum(len(i['iterations']) for i in log['interactions'])
    total_tool_calls = sum(i.get('total_tool_calls', 0) for i in log['interactions'])
    
    # Token usage if available
    usage = log['interactions'][-1]['iterations'][-1]['model_info']['usage']
    total_input_tokens = usage.get('input_tokens', 0)
    total_output_tokens = usage.get('output_tokens', 0)
    
    print("─" * 80)
    print(f"Agents: {len(log['interactions'])} | "
          f"Iterations: {total_iterations} | "
          f"Tool calls: {total_tool_calls}")
    
    if total_input_tokens > 0 or total_output_tokens > 0:
        total_tokens = total_input_tokens + total_output_tokens
        print(f"Tokens: {total_tokens:,} total ({total_input_tokens:,} in, {total_output_tokens:,} out)")
    
    print("─" * 80)


def print_tree(log: Dict):
    """Print a tree view of agent hierarchy."""
    print("=" * 80)
    print("Agent Call Tree")
    print("=" * 80)
    print()
    
    # Build parent-child relationships
    interactions_by_agent = {i['agent']: i for i in log['interactions']}
    root_interactions = [i for i in log['interactions'] if not i.get('parent')]
    
    def print_interaction(interaction, depth=0):
        indent = "  " * depth
        symbol = "└─" if depth > 0 else "●"
        
        print(f"{indent}{symbol} {interaction['agent']}")
        print(f"{indent}   ├─ Iterations: {len(interaction['iterations'])}")
        print(f"{indent}   ├─ Tools: {interaction.get('total_tool_calls', 0)}")
        print(f"{indent}   └─ Time: {interaction.get('duration', 0):.2f}s")
        
        # Find children
        children = [i for i in log['interactions'] if i.get('parent') == interaction['agent']]
        for child in children:
            print_interaction(child, depth + 1)
    
    for interaction in root_interactions:
        print_interaction(interaction)
        print()


def print_summary(log: Dict):
    """Print summary statistics."""
    total_iterations = sum(len(i['iterations']) for i in log['interactions'])
    total_tool_calls = sum(i.get('total_tool_calls', 0) for i in log['interactions'])
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total duration: {log['duration']:.2f}s")
    print(f"Agents invoked: {len(log['interactions'])}")
    print(f"Total iterations: {total_iterations}")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Total token count: {log['interactions'][-1]['iterations'][-1]['model_info']['usage'].get('total_tokens', 'N/A')}")
    print()
    
    # Per-agent breakdown
    print("Per-agent breakdown:")
    for interaction in log['interactions']:
        print(f"  {interaction['agent']:<20} {len(interaction['iterations'])} iters, "
              f"{interaction.get('total_tool_calls', 0)} tools, "
              f"{interaction.get('duration', 0):.2f}s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_log.py <log_file.json> [--tree|--summary]")
        print()
        print("Options:")
        print("  --tree     Show agent call tree")
        print("  --summary  Show summary statistics")
        print("  (default)  Show timeline view")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    log = load_log(log_file)
    
    mode = sys.argv[2] if len(sys.argv) > 2 else "timeline"
    
    if mode == "--tree":
        print_tree(log)
    elif mode == "--summary":
        print_summary(log)
    else:
        print_timeline(log)


if __name__ == "__main__":
    main()