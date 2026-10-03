#!/usr/bin/env bash
# Create .venv and install dependencies. Needs Python 3.11 or 3.12 (set PYTHON=python3.12 to choose one).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or newer is required; set PYTHON=python3.12")'
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Setup done. Next: ./run_all.sh"
