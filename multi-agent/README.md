# LinguaHebrew — Multi-Agent Builder

> **15 AI agents** collaborate to build a full-stack Hebrew language learning app — automatically.

---

## What It Builds

**LinguaHebrew** — Learn any language starting from Hebrew.

| Feature | Agent Responsible |
|---|---|
| Landing page + base scaffold | Foundation Agent |
| Beautiful UI components + Dashboard | UI/Design Agent |
| Vocabulary flashcards + spaced repetition | Vocabulary Agent |
| Grammar lessons + exercises | Grammar Agent |
| AI conversation tutor (Claude-powered) | AI Tutor Agent |
| Quiz engine (multiple types) | Quiz Agent |
| XP, streaks, achievements | Progress Agent |
| Frontend code review | Frontend QA Agent |
| API/backend review | Backend QA Agent |
| UX/accessibility audit | UX QA Agent |
| Cross-feature integration check | Integration QA Agent |
| Git branch merging | Integration Agent |
| Build validation + npm build | Deploy Agent |
| Overall monitoring + failure recovery | Monitor Agent |

**Total: 15 agents**

---

## Architecture

```
orchestrator.py
│
├── Phase 0: Foundation Agent ──────────────── (sequential)
│   └── Builds Next.js scaffold, commits to main
│
├── Phase 1: Feature + QA Agents ───────────── (parallel)
│   ├── feature-ui        ←→ qa-frontend
│   ├── feature-vocabulary ←→ qa-frontend
│   ├── feature-grammar   ←→ qa-frontend
│   ├── feature-ai-tutor  ←→ qa-frontend + qa-backend
│   ├── feature-quiz      ←→ qa-frontend
│   ├── feature-progress  ←→ qa-frontend
│   ├── qa-ux             (reviews all pages)
│   └── qa-integration    (waits for all approvals)
│
├── Phase 2: Integration Agent ─────────────── (sequential)
│   └── Merges feature/* branches into main
│
└── Phase 3: Deploy + Monitor ──────────────── (parallel)
    ├── deploy-agent (npm install + npm run build)
    └── monitor-agent (watches all, handles failures)
```

### Inter-Agent Communication

Agents communicate via a **file-based message bus** (`agent_bus/messages/`).
Each message is a JSON file: `{to_agent}_{timestamp}.json`

Message types:
- `ready_for_qa` — feature → QA
- `bug_report` — QA → feature (with list of issues)
- `fix_complete` — feature → QA
- `qa_approved` — QA → feature + integration-QA
- `ready_to_deploy` — integration → deploy
- `deployment_complete` — deploy → monitor

### Git Branches

```
main                  ← foundation + final merge
feature/ui-design
feature/vocabulary
feature/grammar
feature/ai-tutor
feature/quiz
feature/progress
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the built app)
- Anthropic API key

### Setup

```bash
cd multi-agent

# Copy and edit .env
cp .env.example .env
# Add your key: ANTHROPIC_API_KEY=sk-ant-...

# Run the full pipeline
./run.sh
```

### Options

```bash
# Skip foundation (if already built)
./run.sh --skip-foundation

# Run only a specific phase
./run.sh --phase 0    # foundation only
./run.sh --phase 1    # features + QA only
./run.sh --phase 2    # integration only
./run.sh --phase 3    # deploy + monitor only
```

### After the Pipeline

```bash
cd lang-app
npm install
npm run dev
# Open http://localhost:3000
```

---

## Cost Estimate

Each agent uses `claude-sonnet-4-6` with up to 60 iterations.
Estimated total: **~$5–15** depending on code complexity.

For cheaper runs: edit `agents/roles.py` and set `model = "claude-haiku-4-5-20251001"`.

---

## File Structure

```
multi-agent/
├── orchestrator.py        # Main entry point
├── run.sh                 # Convenience runner
├── requirements.txt       # anthropic SDK
├── .env.example
├── config/
│   └── agents.json        # Agent + app configuration
├── agents/
│   ├── __init__.py
│   ├── base_agent.py      # Async agentic loop
│   ├── tools.py           # Tool implementations + schemas
│   └── roles.py           # All 15 agent role definitions
├── bus/
│   ├── __init__.py
│   └── message_bus.py     # File-based inter-agent bus
├── agent_bus/
│   ├── messages/          # In-flight messages
│   └── status/            # Agent status files
├── logs/                  # Run logs
└── lang-app/              # The built application (output)
```

---

## The App: LinguaHebrew

**Tech Stack**
- Next.js 14 (App Router, TypeScript)
- TailwindCSS (mobile-first, deep blue + gold palette)
- Anthropic Claude API for AI tutor (streaming)
- lucide-react icons

**Pages**
- `/` — Landing page
- `/dashboard` — User dashboard (progress, streaks)
- `/vocabulary` → `/vocabulary/[category]` → quiz
- `/grammar` → `/grammar/[topic]`
- `/tutor` → `/tutor/[scenario]` (AI conversation)
- `/quiz` → `/quiz/[type]`
- `/progress` — Full progress breakdown
- `/settings` — Language + preferences

**Supported Languages** (Hebrew → target):
English, Arabic, French, Spanish, Russian, German, Chinese, Japanese, Portuguese, Italian, Dutch, Turkish, Polish, Korean
