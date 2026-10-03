@echo off
rem Create .venv and install dependencies. Needs Python 3.11 or 3.12 from python.org.
cd /d %~dp0
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo Setup done. Next: run_all.bat
