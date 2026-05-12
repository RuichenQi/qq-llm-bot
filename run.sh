#!/usr/bin/env bash
# Single-command runner. Use this on Termux or any POSIX shell.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "[run.sh] creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

echo "[run.sh] installing requirements"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "[run.sh] no .env found — copying from .env.example. Edit it and rerun."
  cp .env.example .env
  exit 1
fi

echo "[run.sh] starting bot"
exec python main.py
