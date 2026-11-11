@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   Campus RAG QA Agent - demo launcher
echo   -----------------------------------
"%PY%" demo\start_demo.py %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo   [x] Startup failed with code %CODE%. Check the messages above.
  pause
)
