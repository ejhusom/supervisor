"""HDFS log utilities for session extraction."""

import re


def extract_hdfs_sessions(log_content: str, max_sessions: int = 10) -> dict:
    """
    Extract HDFS block sessions from log content.

    Groups log lines by their block ID (blk_*) into sessions.

    Args:
        log_content: Raw HDFS log content as a string.
        max_sessions: Maximum number of sessions to return (0 = unlimited).

    Returns:
        Dictionary mapping block IDs to lists of log lines:
        {
            "blk_123...": ["log line 1", "log line 2", ...],
            "blk_456...": ["log line 1", ...],
        }
    """
    sessions = {}
    pattern = r"blk_-?\d+"

    for line in log_content.strip().split("\n"):
        match = re.search(pattern, line)
        if match:
            blk_id = match.group()
            if blk_id not in sessions:
                sessions[blk_id] = []
            sessions[blk_id].append(line)

    if max_sessions and len(sessions) > max_sessions:
        sessions = dict(list(sessions.items())[:max_sessions])

    return sessions


def format_session_for_analysis(blk_id: str, lines: list[str]) -> str:
    """
    Format a single session for LLM analysis.

    Args:
        blk_id: Block ID.
        lines: List of log lines for this block.

    Returns:
        Formatted string ready for analysis.
    """
    return f"=== Block Session: {blk_id} ({len(lines)} events) ===\n" + "\n".join(
        lines
    )
