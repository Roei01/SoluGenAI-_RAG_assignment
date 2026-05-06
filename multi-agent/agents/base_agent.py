"""
BaseAgent: async agentic loop using the Anthropic API with tool use.
Each agent runs independently, processes tool calls, and communicates via the bus.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import anthropic

from bus import MessageBus, AgentStatus
from agents.tools import (
    TOOL_SCHEMAS, execute_command, write_file, read_file,
    list_files, git_commit, git_checkout, git_merge, git_status,
)

logger = logging.getLogger("agent")


class BaseAgent:
    """
    Async agentic loop.  Each iteration:
      1. Builds context (messages in bus + current status)
      2. Calls Claude API
      3. Executes any tool calls
      4. Loops until task_complete / task_failed / max_iterations
    """

    MAX_ITERATIONS = 60

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        app_dir: Path,
        bus: MessageBus,
        status: AgentStatus,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.app_dir = app_dir
        self.bus = bus
        self.status = status
        self.client = client
        self.model = model
        self.log = logging.getLogger(f"agent.{name}")
        self._done = False
        self._success = False
        self._summary = ""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def run(self, initial_task: str) -> bool:
        """Run the agent.  Returns True on success."""
        self.status.update(self.name, "working", 0, "Starting…")
        self.log.info(f"[{self.name}] Starting: {initial_task[:80]}")

        messages: list[dict] = [
            {"role": "user", "content": initial_task}
        ]

        for iteration in range(self.MAX_ITERATIONS):
            if self._done:
                break

            # Inject any pending bus messages as a user turn
            inbox = self.bus.receive(self.name)
            if inbox and messages[-1]["role"] == "assistant":
                summary = "\n".join(
                    f"[MSG from {m['from']} | {m['type']}]: {json.dumps(m['content'], ensure_ascii=False)}"
                    for m in inbox
                )
                messages.append({"role": "user", "content": f"You received these messages:\n{summary}\n\nContinue your work."})

            try:
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=8096,
                    system=self.system_prompt,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except anthropic.RateLimitError:
                self.log.warning(f"[{self.name}] Rate limited, sleeping 30s")
                await asyncio.sleep(30)
                continue
            except Exception as e:
                self.log.error(f"[{self.name}] API error: {e}")
                await asyncio.sleep(10)
                continue

            # Append assistant response
            messages.append({"role": "assistant", "content": response.content})

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._handle_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Check stop condition
            if response.stop_reason == "end_turn" and not tool_results:
                # Model finished without calling tools — treat as done
                self.log.info(f"[{self.name}] Finished (end_turn, no tools)")
                self._done = True
                self._success = True
                self.status.update(self.name, "done", 100, "Completed")
                break

            progress = min(95, int(iteration / self.MAX_ITERATIONS * 100))
            self.status.update(self.name, "working", progress, f"Iteration {iteration+1}")
            await asyncio.sleep(0.5)  # small yield to other agents

        if not self._done:
            self.log.warning(f"[{self.name}] Reached max iterations")
            self.status.update(self.name, "failed", 0, "Max iterations reached")
            return False

        return self._success

    # ------------------------------------------------------------------ #
    # Tool dispatch
    # ------------------------------------------------------------------ #

    async def _handle_tool(self, name: str, inputs: dict) -> Any:
        self.log.info(f"[{self.name}] TOOL {name}({list(inputs.keys())})")

        if name == "execute_command":
            return execute_command(inputs["command"], self.app_dir, inputs.get("timeout", 120))

        elif name == "write_file":
            return write_file(inputs["path"], inputs["content"], self.app_dir)

        elif name == "read_file":
            return read_file(inputs["path"], self.app_dir)

        elif name == "list_files":
            return list_files(inputs.get("directory", "."), self.app_dir)

        elif name == "git_commit":
            return git_commit(inputs["message"], self.app_dir)

        elif name == "git_checkout":
            return git_checkout(inputs["branch"], inputs.get("create", False), self.app_dir)

        elif name == "git_merge":
            return git_merge(inputs["branch"], self.app_dir)

        elif name == "git_status":
            return git_status(self.app_dir)

        elif name == "send_message":
            self.bus.send(self.name, inputs["to_agent"], inputs["msg_type"], inputs["content"])
            return {"success": True}

        elif name == "get_messages":
            msgs = self.bus.receive(self.name)
            return {"messages": msgs, "count": len(msgs)}

        elif name == "task_complete":
            self._done = True
            self._success = True
            self._summary = inputs.get("summary", "")
            self.status.update(self.name, "done", 100, self._summary[:200])
            self.log.info(f"[{self.name}] DONE: {self._summary[:100]}")
            return {"acknowledged": True}

        elif name == "task_failed":
            self._done = True
            self._success = False
            self.status.update(self.name, "failed", 0, inputs.get("reason", ""))
            self.log.error(f"[{self.name}] FAILED: {inputs.get('reason', '')}")
            return {"acknowledged": True}

        else:
            return {"error": f"Unknown tool: {name}"}
