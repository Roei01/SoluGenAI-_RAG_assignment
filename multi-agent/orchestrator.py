#!/usr/bin/env python3
"""
LinguaHebrew Multi-Agent Orchestrator
======================================
Coordinates 15 AI agents (Claude instances) to build a full-stack
Hebrew language learning web application.

Pipeline:
  Phase 0 — Foundation Agent builds the base Next.js app
  Phase 1 — 6 Feature Agents work in parallel (each on its own branch)
           — 4 QA Agents run concurrently (review feature branches)
  Phase 2 — Integration Agent merges all branches
  Phase 3 — Deploy Agent validates & builds the app
           — Monitor Agent watches over the entire run

Usage:
  python orchestrator.py [--skip-foundation] [--phase PHASE]
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic

# Ensure we can import from siblings
sys.path.insert(0, str(Path(__file__).parent))

from bus import MessageBus, AgentStatus
from agents import BaseAgent, AGENT_ROLES, FEATURE_AGENTS, QA_AGENTS, PIPELINE_AGENTS, build_task

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

def setup_logging(logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s [%(name)-22s] %(levelname)-7s %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / f"run_{ts}.log"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Quieter third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #

BASE_DIR   = Path(__file__).parent
APP_DIR    = BASE_DIR / "lang-app"
BUS_DIR    = BASE_DIR / "agent_bus"
MSG_DIR    = BUS_DIR / "messages"
STATUS_DIR = BUS_DIR / "status"
LOGS_DIR   = BASE_DIR / "logs"

for d in [APP_DIR, MSG_DIR, STATUS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Git bootstrap
# ------------------------------------------------------------------ #

def run(cmd: str, cwd: Path = APP_DIR, check: bool = False) -> tuple[str, str, int]:
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def init_git_repo():
    log = logging.getLogger("orchestrator")
    if not (APP_DIR / ".git").exists():
        run("git init")
        run('git config user.email "agents@linguahebrew.ai"')
        run('git config user.name "LinguaHebrew Agents"')
        # Placeholder commit so branches can be created
        (APP_DIR / ".gitkeep").write_text("# LinguaHebrew\n")
        run("git add .gitkeep")
        run('git commit -m "chore: init repo"')
        log.info("Git repository initialized")
    else:
        log.info("Git repository already exists")


# ------------------------------------------------------------------ #
# Dashboard printer
# ------------------------------------------------------------------ #

PHASE_EMOJI = {
    "idle":    "⬜",
    "working": "🔄",
    "waiting": "⏳",
    "done":    "✅",
    "failed":  "❌",
}

def print_dashboard(status: AgentStatus, phase: str):
    all_agents = (
        ["foundation"]
        + FEATURE_AGENTS
        + QA_AGENTS
        + PIPELINE_AGENTS
    )
    statuses = status.all()
    print(f"\n{'='*64}")
    print(f"  LINGUAHEBREW AGENTS — Phase: {phase}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*64}")
    for name in all_agents:
        s = statuses.get(name, {})
        emoji = PHASE_EMOJI.get(s.get("status", "idle"), "⬜")
        pct   = s.get("progress", 0)
        detail = s.get("details", "")[:40]
        role  = AGENT_ROLES[name]["role"]
        bar   = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {emoji} {name:<22} [{bar}] {pct:3d}%  {detail}")
    print(f"{'='*64}\n")


# ------------------------------------------------------------------ #
# Agent runner factory
# ------------------------------------------------------------------ #

def make_agent(name: str, client: anthropic.Anthropic, bus: MessageBus, status: AgentStatus) -> BaseAgent:
    cfg = AGENT_ROLES[name]
    return BaseAgent(
        name=name,
        role=cfg["role"],
        system_prompt=cfg["system_prompt"],
        app_dir=APP_DIR,
        bus=bus,
        status=status,
        client=client,
        model=cfg.get("model", "claude-sonnet-4-6"),
    )


async def run_agent(name: str, task: str, client: anthropic.Anthropic,
                    bus: MessageBus, status: AgentStatus) -> bool:
    agent = make_agent(name, client, bus, status)
    try:
        return await agent.run(task)
    except Exception as e:
        logging.getLogger(f"agent.{name}").error(f"Unhandled exception: {e}", exc_info=True)
        status.update(name, "failed", 0, str(e)[:200])
        return False


# ------------------------------------------------------------------ #
# Phase runners
# ------------------------------------------------------------------ #

async def phase0_foundation(client, bus, status, log) -> bool:
    """Run Foundation Agent — must finish before anything else."""
    log.info("━━━ PHASE 0: Foundation ━━━")
    status.update("foundation", "working", 0, "Starting foundation build")

    task = build_task(
        "foundation",
        "Build the complete Next.js scaffold for LinguaHebrew. "
        "Write ALL files listed in your instructions. "
        "After writing all files, run: git add -A && git commit -m 'feat: foundation scaffold'"
    )

    success = await run_agent("foundation", task, client, bus, status)
    if success:
        log.info("✅ Foundation complete!")
    else:
        log.error("❌ Foundation FAILED — check logs")
    return success


async def phase1_features_and_qa(client, bus, status, log):
    """Run all 6 feature agents + 4 QA agents in parallel."""
    log.info("━━━ PHASE 1: Features + QA (parallel) ━━━")

    # Notify QA agents to start monitoring
    for qa in QA_AGENTS:
        status.update(qa, "waiting", 0, "Waiting for feature branches")

    feature_tasks = []
    for name in FEATURE_AGENTS:
        task = build_task(
            name,
            f"The foundation is ready on main. "
            f"Checkout your branch, build your feature, communicate with qa-frontend."
        )
        feature_tasks.append(run_agent(name, task, client, bus, status))

    qa_tasks = []
    for name in QA_AGENTS:
        task = build_task(
            name,
            "Feature agents are now running. Check your bus for ready_for_qa messages. "
            "Review branches as they become available."
        )
        qa_tasks.append(run_agent(name, task, client, bus, status))

    all_tasks = feature_tasks + qa_tasks
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    successes = sum(1 for r in results if r is True)
    failures  = sum(1 for r in results if r is not True)
    log.info(f"Phase 1 complete: {successes} succeeded, {failures} failed")
    return failures == 0


async def phase2_integration(client, bus, status, log) -> bool:
    """Merge all feature branches."""
    log.info("━━━ PHASE 2: Integration ━━━")
    task = build_task(
        "integration-agent",
        "All feature branches are ready. Merge them into main and finalize the app."
    )
    success = await run_agent("integration-agent", task, client, bus, status)
    log.info("✅ Integration complete!" if success else "❌ Integration FAILED")
    return success


async def phase3_deploy_and_monitor(client, bus, status, log):
    """Deploy validation + monitor + recovery run concurrently."""
    log.info("━━━ PHASE 3: Deploy + Monitor + Recovery ━━━")

    deploy_task = build_task(
        "deploy-agent",
        "Integration is complete. Build and validate the final app."
    )
    monitor_task = build_task(
        "monitor-agent",
        "The full pipeline is running. Monitor all agents and handle any failures. "
        "Finalize when deploy-agent sends deployment_complete."
    )
    recovery_task = build_task(
        "recovery-agent",
        "Stand by. If the deploy-agent or monitor-agent sends you a 'recovery_needed' "
        "message, diagnose and fix the broken state. Otherwise, do a proactive audit: "
        "list all files, check for common issues (missing files, broken imports), and fix them."
    )

    results = await asyncio.gather(
        run_agent("deploy-agent", deploy_task, client, bus, status),
        run_agent("monitor-agent", monitor_task, client, bus, status),
        run_agent("recovery-agent", recovery_task, client, bus, status),
        return_exceptions=True,
    )
    success = all(r is True for r in results)
    log.info("✅ Deploy + Monitor complete!" if success else "⚠️  Some issues in deploy phase")
    return success


# ------------------------------------------------------------------ #
# Dashboard loop (background)
# ------------------------------------------------------------------ #

async def dashboard_loop(status: AgentStatus, phase_ref: list, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            print_dashboard(status, phase_ref[0])
        except Exception:
            pass
        await asyncio.sleep(30)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

async def main(args):
    setup_logging(LOGS_DIR)
    log = logging.getLogger("orchestrator")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable not set!")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    bus    = MessageBus(MSG_DIR)
    status = AgentStatus(STATUS_DIR)

    log.info("🚀 LinguaHebrew Multi-Agent System Starting")
    log.info(f"   App dir:  {APP_DIR}")
    log.info(f"   Agents:   15 (1 foundation + 6 feature + 4 QA + 3 pipeline)")
    log.info(f"   Model:    claude-sonnet-4-6")

    # Init git
    if not args.skip_git:
        init_git_repo()

    phase_ref = ["starting"]
    stop_event = asyncio.Event()

    # Start dashboard in background
    dashboard_task = asyncio.create_task(dashboard_loop(status, phase_ref, stop_event))

    start = time.time()
    overall_success = True

    try:
        # ── Phase 0: Foundation ──────────────────────────────────────
        if not args.skip_foundation:
            phase_ref[0] = "Phase 0: Foundation"
            ok = await phase0_foundation(client, bus, status, log)
            if not ok:
                log.error("Foundation failed — aborting pipeline")
                overall_success = False
                return
        else:
            log.info("Skipping foundation (--skip-foundation)")
            status.update("foundation", "done", 100, "Skipped")

        # ── Phase 1: Features + QA ───────────────────────────────────
        if args.phase in (None, 1, "1", "all"):
            phase_ref[0] = "Phase 1: Features + QA"
            await phase1_features_and_qa(client, bus, status, log)

        # ── Phase 2: Integration ─────────────────────────────────────
        if args.phase in (None, 2, "2", "all"):
            phase_ref[0] = "Phase 2: Integration"
            ok = await phase2_integration(client, bus, status, log)
            if not ok:
                overall_success = False

        # ── Phase 3: Deploy + Monitor ────────────────────────────────
        if args.phase in (None, 3, "3", "all"):
            phase_ref[0] = "Phase 3: Deploy + Monitor"
            ok = await phase3_deploy_and_monitor(client, bus, status, log)
            if not ok:
                overall_success = False

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Pipeline error: {e}", exc_info=True)
        overall_success = False
    finally:
        stop_event.set()
        dashboard_task.cancel()
        elapsed = time.time() - start

    # Final report
    phase_ref[0] = "COMPLETE" if overall_success else "FAILED"
    print_dashboard(status, phase_ref[0])

    log.info(f"\n{'='*64}")
    log.info(f"  PIPELINE {'COMPLETE ✅' if overall_success else 'FAILED ❌'}")
    log.info(f"  Elapsed: {elapsed/60:.1f} minutes")
    log.info(f"  App dir: {APP_DIR}")
    if overall_success:
        log.info("  To run: cd lang-app && npm run dev")
        log.info("  Open:   http://localhost:3000")
    log.info(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinguaHebrew Multi-Agent Orchestrator")
    parser.add_argument("--skip-foundation", action="store_true",
                        help="Skip foundation agent (useful if already built)")
    parser.add_argument("--skip-git", action="store_true",
                        help="Skip git repo initialization")
    parser.add_argument("--phase", default=None,
                        help="Run only a specific phase (0, 1, 2, 3, or all)")
    args = parser.parse_args()

    asyncio.run(main(args))
