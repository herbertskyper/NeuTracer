"""
AI Reasoning Process Logger
Collects and analyzes AI reasoning processes for better understanding and debugging.
"""

import json
import os
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class ReasoningLogger:
    """Logs and analyzes AI reasoning processes."""

    def __init__(self, log_dir: str = None):
        """
        Initialize the reasoning logger.

        Args:
            log_dir (str, optional): Directory to store logs. Defaults to ~/.llm-cli-tool/logs
        """
        if log_dir is None:
            self.log_dir = Path.home() / ".llm-cli-tool" / "reasoning_logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id = self._generate_session_id()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def log_reasoning_step(
        self, step_type: str, content: str, metadata: Dict[str, Any] = None
    ) -> None:
        """
        Log a single reasoning step.

        Args:
            step_type (str): Type of reasoning step (e.g., 'input', 'analysis', 'conclusion')
            content (str): The content of this reasoning step
            metadata (dict, optional): Additional metadata for this step
        """
        log_entry = {
            "session_id": self.current_session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "step_type": step_type,
            "content": content,
            "metadata": metadata or {},
        }

        log_file = self.log_dir / f"reasoning_{self.current_session_id}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def start_new_session(self, context: Dict[str, Any] = None) -> str:
        """
        Start a new reasoning session.

        Args:
            context (dict, optional): Initial context for the session

        Returns:
            str: The new session ID
        """
        self.current_session_id = self._generate_session_id()

        if context:
            self.log_reasoning_step(
                "session_start", "New reasoning session started", context
            )

        return self.current_session_id

    def end_session(self, summary: str = None) -> None:
        """
        End the current reasoning session.

        Args:
            summary (str, optional): Summary of the session
        """
        self.log_reasoning_step("session_end", summary or "Session ended")

    def get_session_logs(self, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Get logs for a specific session.

        Args:
            session_id (str, optional): Session ID. Defaults to current session.

        Returns:
            List[Dict[str, Any]]: List of log entries
        """
        if session_id is None:
            session_id = self.current_session_id

        log_file = self.log_dir / f"reasoning_{session_id}.jsonl"

        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))

        return logs

    def list_sessions(self) -> List[str]:
        """
        List all available reasoning sessions.

        Returns:
            List[str]: List of session IDs
        """
        session_files = list(self.log_dir.glob("reasoning_*.jsonl"))
        session_ids = [f.stem.replace("reasoning_", "") for f in session_files]
        return sorted(session_ids, reverse=True)  # Most recent first

    def analyze_session(self, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze a reasoning session.

        Args:
            session_id (str, optional): Session ID. Defaults to current session.

        Returns:
            Dict[str, Any]: Analysis results
        """
        logs = self.get_session_logs(session_id)

        if not logs:
            return {"error": "No logs found for session"}

        analysis = {
            "session_id": session_id or self.current_session_id,
            "total_steps": len(logs),
            "step_types": {},
            "duration": None,
            "start_time": None,
            "end_time": None,
        }

        # Count step types
        for log in logs:
            step_type = log.get("step_type", "unknown")
            analysis["step_types"][step_type] = (
                analysis["step_types"].get(step_type, 0) + 1
            )

        # Calculate duration
        if logs:
            start_time = datetime.datetime.fromisoformat(logs[0]["timestamp"])
            end_time = datetime.datetime.fromisoformat(logs[-1]["timestamp"])
            analysis["start_time"] = start_time.isoformat()
            analysis["end_time"] = end_time.isoformat()
            analysis["duration"] = (end_time - start_time).total_seconds()

        return analysis


def read_file_content(file_path: str) -> str:
    """
    Read content from a file.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: File content

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def prepare_file_analysis_prompt(file_path: str, question: str = None) -> str:
    """
    Prepare a prompt for analyzing a file.

    Args:
        file_path (str): Path to the file to analyze
        question (str, optional): Specific question about the file

    Returns:
        str: Formatted prompt for the AI
    """
    try:
        content = read_file_content(file_path)
        file_info = {
            "name": os.path.basename(file_path),
            "size": len(content),
            "lines": content.count("\n") + 1,
        }

        prompt = f"""请分析以下文件内容：

文件信息:
- 文件名: {file_info['name']}
- 大小: {file_info['size']} 字符
- 行数: {file_info['lines']} 行

文件内容:
```
{content}
```

"""

        if question:
            prompt += f"\n具体问题: {question}\n"
        else:
            prompt += (
                "\n请提供对这个文件的详细分析，包括其结构、功能和可能的改进建议。\n"
            )

        prompt += "\n请提供详细的推理过程，包括你的分析步骤和结论。"

        return prompt

    except Exception as e:
        return f"无法读取文件 {file_path}: {str(e)}"
