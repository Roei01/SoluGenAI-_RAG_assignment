"""
File-based inter-agent message bus.
Each message is a JSON file: {to_agent}_{timestamp}.json
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bus")


class MessageBus:
    def __init__(self, messages_dir: Path):
        self.dir = messages_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, from_agent: str, to_agent: str, msg_type: str, content: dict):
        msg = {
            "id": f"{time.time():.6f}",
            "from": from_agent,
            "to": to_agent,
            "type": msg_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        path = self.dir / f"{to_agent}_{msg['id']}.json"
        path.write_text(json.dumps(msg, ensure_ascii=False, indent=2))
        logger.debug(f"  BUS  {from_agent} → {to_agent} [{msg_type}]")

    def receive(self, agent_name: str) -> list[dict]:
        msgs = []
        for f in sorted(self.dir.glob(f"{agent_name}_*.json")):
            try:
                msgs.append(json.loads(f.read_text()))
                f.unlink()
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
        return msgs

    def broadcast(self, from_agent: str, msg_type: str, content: dict, targets: list[str]):
        for t in targets:
            self.send(from_agent, t, msg_type, content)


class AgentStatus:
    def __init__(self, status_dir: Path):
        self.dir = status_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def update(self, agent_name: str, status: str, progress: int = 0, details: str = ""):
        data = {
            "agent": agent_name,
            "status": status,      # idle | working | done | failed | waiting
            "progress": progress,  # 0-100
            "details": details,
            "updated_at": datetime.now().isoformat(),
        }
        (self.dir / f"{agent_name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )

    def get(self, agent_name: str) -> Optional[dict]:
        p = self.dir / f"{agent_name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def all(self) -> dict[str, dict]:
        result = {}
        for f in self.dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                result[d["agent"]] = d
            except Exception:
                pass
        return result

    def all_done(self, agents: list[str]) -> bool:
        for a in agents:
            s = self.get(a)
            if not s or s["status"] not in ("done", "failed"):
                return False
        return True
