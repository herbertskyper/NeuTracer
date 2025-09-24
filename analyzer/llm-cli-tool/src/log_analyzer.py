"""
Reasoning Log Analysis Tool
Provides detailed analysis of AI reasoning logs.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from .reasoning_logger import ReasoningLogger


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def display_session_summary(session_id: str, logger: ReasoningLogger) -> None:
    """Display a summary of a reasoning session."""
    analysis = logger.analyze_session(session_id)
    logs = logger.get_session_logs(session_id)

    if not logs:
        print(f"No logs found for session {session_id}")
        return

    print(f"Session ID: {session_id}")
    print("=" * 50)
    print(f"Start Time: {format_timestamp(analysis['start_time'])}")
    print(f"End Time: {format_timestamp(analysis['end_time'])}")
    print(f"Duration: {analysis['duration']:.2f} seconds")
    print(f"Total Steps: {analysis['total_steps']}")
    print()

    print("Step Types:")
    for step_type, count in analysis["step_types"].items():
        print(f"  {step_type}: {count}")
    print()


def display_detailed_logs(
    session_id: str, logger: ReasoningLogger, max_content_length: int = 200
) -> None:
    """Display detailed logs for a session."""
    logs = logger.get_session_logs(session_id)

    if not logs:
        print(f"No logs found for session {session_id}")
        return

    print(f"Detailed Logs for Session: {session_id}")
    print("=" * 60)

    for i, log in enumerate(logs, 1):
        timestamp = format_timestamp(log["timestamp"])
        step_type = log["step_type"]
        content = log["content"]

        # Truncate content if too long
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        print(f"Step {i}: [{timestamp}] {step_type}")
        print(f"Content: {content}")

        if log.get("metadata"):
            print(f"Metadata: {json.dumps(log['metadata'], indent=2)}")

        print("-" * 40)


def export_session(session_id: str, logger: ReasoningLogger, output_file: str) -> None:
    """Export session logs to a file."""
    logs = logger.get_session_logs(session_id)
    analysis = logger.analyze_session(session_id)

    export_data = {"session_id": session_id, "analysis": analysis, "logs": logs}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Session {session_id} exported to {output_file}")


def main():
    """Main function for the log analysis tool."""
    parser = argparse.ArgumentParser(description="Analyze AI reasoning logs")

    parser.add_argument(
        "session_id",
        nargs="?",
        help="Session ID to analyze (if not provided, lists all sessions)",
    )

    parser.add_argument(
        "--log-dir", type=str, help="Directory containing reasoning logs"
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed logs instead of just summary",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="OUTPUT_FILE",
        help="Export session data to a JSON file",
    )

    parser.add_argument(
        "--max-content",
        type=int,
        default=200,
        help="Maximum content length to display (default: 200)",
    )

    args = parser.parse_args()

    logger = ReasoningLogger(args.log_dir)

    if not args.session_id:
        # List all sessions
        sessions = logger.list_sessions()
        if sessions:
            print("Available reasoning sessions:")
            print("-" * 30)
            for session in sessions:
                print(f"  {session}")
        else:
            print("No reasoning sessions found.")
        return

    # Analyze specific session
    if args.export:
        export_session(args.session_id, logger, args.export)
    elif args.detailed:
        display_detailed_logs(args.session_id, logger, args.max_content)
    else:
        display_session_summary(args.session_id, logger)


if __name__ == "__main__":
    main()
