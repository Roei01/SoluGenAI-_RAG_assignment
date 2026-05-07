"""
Tool implementations executed on behalf of agents.
Each tool is called by the agent's async loop when Claude requests it.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def execute_command(command: str, cwd: Path, timeout: int = 120) -> dict:
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CI": "true", "FORCE_COLOR": "0"},
        )
        return {
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1, "success": False}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


def write_file(path: str, content: str, app_dir: Path) -> dict:
    try:
        target = (app_dir / path).resolve()
        # Safety: must stay within app_dir
        if not str(target).startswith(str(app_dir.resolve())):
            return {"success": False, "error": "Path outside app directory"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(target.relative_to(app_dir))}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(path: str, app_dir: Path) -> dict:
    try:
        target = (app_dir / path).resolve()
        if not str(target).startswith(str(app_dir.resolve())):
            return {"success": False, "error": "Path outside app directory"}
        if not target.exists():
            return {"success": False, "error": f"File not found: {path}"}
        content = target.read_text(encoding="utf-8")
        return {"success": True, "content": content[:8000], "truncated": len(content) > 8000}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_files(directory: str, app_dir: Path) -> dict:
    try:
        target = (app_dir / directory).resolve()
        if not str(target).startswith(str(app_dir.resolve())):
            return {"success": False, "error": "Path outside app directory"}
        if not target.exists():
            return {"success": False, "error": "Directory not found"}
        files = []
        for p in sorted(target.rglob("*")):
            if p.is_file() and ".git" not in p.parts and "node_modules" not in p.parts:
                files.append(str(p.relative_to(app_dir)))
        return {"success": True, "files": files[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def git_commit(message: str, cwd: Path) -> dict:
    r1 = execute_command("git add -A", cwd)
    r2 = execute_command(f'git commit -m "{message}" --allow-empty', cwd)
    return {"success": r2["success"], "stdout": r2["stdout"], "stderr": r2["stderr"]}


def git_checkout(branch: str, create: bool, cwd: Path) -> dict:
    flag = "-b" if create else ""
    return execute_command(f"git checkout {flag} {branch}", cwd)


def git_merge(branch: str, cwd: Path) -> dict:
    return execute_command(f"git merge {branch} --no-ff -m 'Merge {branch}'", cwd)


def git_status(cwd: Path) -> dict:
    r = execute_command("git status --short", cwd)
    r2 = execute_command("git branch --show-current", cwd)
    return {
        "success": True,
        "status": r["stdout"],
        "branch": r2["stdout"].strip(),
    }


# Anthropic tool schemas for the agents
TOOL_SCHEMAS = [
    {
        "name": "execute_command",
        "description": "Execute a shell command in the app directory. Use for npm, git, file operations, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (relative to app directory). Creates parent dirs automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the app directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory (relative to app directory).",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path (use '.' for root)"},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "git_commit",
        "description": "Stage all changes and commit to git.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_checkout",
        "description": "Checkout or create a git branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "description": "Branch name"},
                "create": {"type": "boolean", "description": "True to create new branch"},
            },
            "required": ["branch", "create"],
        },
    },
    {
        "name": "git_merge",
        "description": "Merge a branch into the current branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "description": "Branch name to merge"},
            },
            "required": ["branch"],
        },
    },
    {
        "name": "git_status",
        "description": "Get current git status and branch name.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to another agent via the message bus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_agent": {"type": "string", "description": "Target agent name"},
                "msg_type": {"type": "string", "description": "Message type (e.g. bug_report, fix_complete, status_update)"},
                "content": {"type": "object", "description": "Message payload"},
            },
            "required": ["to_agent", "msg_type", "content"],
        },
    },
    {
        "name": "get_messages",
        "description": "Check for incoming messages from other agents.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "task_complete",
        "description": "Signal that this agent has finished its task successfully.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of what was accomplished"},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "task_failed",
        "description": "Signal that this agent failed its task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for failure"},
            },
            "required": ["reason"],
        },
    },
]
