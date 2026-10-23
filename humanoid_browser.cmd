@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first.
  pause
  exit /b 1
)
rem The server runs without a window (log: runs\humanoid_lab\server.log); this window closes at once.
start "" ".venv\Scripts\pythonw.exe" -m humanoid_sim.webapp --open --background %*
