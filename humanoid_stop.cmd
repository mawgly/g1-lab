@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m humanoid_sim.cli stop %*
pause
