#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  run.sh  –  One-click setup & launch for AI Smart Elderly Assistant
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────────
set -e

echo ""
echo "🤖  AI Smart Elderly Assistant – Python + ML"
echo "────────────────────────────────────────────"

# 1. Install dependencies
echo "📦  Installing dependencies…"
pip install -q -r requirements.txt

# 2. Train the ML model (only if not already trained)
if [ ! -f "model/intent_model.pkl" ]; then
  echo "🧠  Training ML model (TF-IDF + Logistic Regression)…"
  python3 train_model.py
else
  echo "✅  ML model already trained – skipping training step."
fi

# 3. Launch Flask server
echo ""
echo "🚀  Starting Flask server at http://localhost:5000"
echo "    Open this URL in Chrome or Edge for full voice support."
echo ""
python3 app.py

