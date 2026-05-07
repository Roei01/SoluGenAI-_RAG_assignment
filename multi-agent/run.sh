#!/usr/bin/env bash
# ─────────────────────────────────────────────
# LinguaHebrew — Multi-Agent Builder
# ─────────────────────────────────────────────
set -e

cd "$(dirname "$0")"

# 1. Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌  python3 not found. Please install Python 3.10+."
  exit 1
fi

# 2. Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅  Loaded .env"
fi

# 3. Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "❌  ANTHROPIC_API_KEY is not set."
  echo "    Copy .env.example to .env and add your key."
  exit 1
fi

# 4. Install Python deps
echo "📦  Installing Python dependencies…"
pip install -q -r requirements.txt

# 5. Create lang-app directory
mkdir -p lang-app

# 6. Run orchestrator
echo ""
echo "🚀  Starting 15-agent pipeline…"
echo "    Logs → ./logs/"
echo "    App  → ./lang-app/"
echo ""

python3 orchestrator.py "$@"

echo ""
echo "✅  Done! To start the app:"
echo "    cd lang-app && npm install && npm run dev"
