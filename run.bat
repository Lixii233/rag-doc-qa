@echo off
rem 启动 Streamlit 界面：双击或在终端执行 run.bat
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo [1/2] 检查依赖...
"%PY%" -c "import streamlit" 2>nul || (
    echo   未安装依赖，正在安装 requirements.txt ...
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo [2/2] 启动界面 http://localhost:8501
"%PY%" -m streamlit run app.py %*
