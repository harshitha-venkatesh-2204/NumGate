@echo off
rem Build tables, run the mock smoke experiment, analyze, and open the report. Costs nothing.
cd /d %~dp0
set PY=.venv\Scripts\python
%PY% src\build_tables.py || exit /b 1
%PY% src\run_experiment.py --smoke || exit /b 1
%PY% src\analyze.py --run-id smoke || exit /b 1
start "" results\dashboard.html
