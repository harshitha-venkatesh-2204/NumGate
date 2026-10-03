#!/usr/bin/env bash
# Build tables, run the mock smoke experiment, analyze, and open the report. Costs nothing.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY src/build_tables.py
$PY src/run_experiment.py --smoke
$PY src/analyze.py --run-id smoke
open results/dashboard.html 2>/dev/null || xdg-open results/dashboard.html 2>/dev/null || echo "Open results/dashboard.html in a browser"
