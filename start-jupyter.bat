@echo off
cd /d "%~dp0"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
set "IPYTHONDIR=%~dp0.jupyter\ipython"
set "JUPYTER_CONFIG_DIR=%~dp0.jupyter"
set "JUPYTER_DATA_DIR=%~dp0.jupyter\data"
set "JUPYTER_RUNTIME_DIR=%~dp0.jupyter\runtime"
"%~dp0.venv\Scripts\python.exe" -m jupyterlab --notebook-dir="%~dp0." %*


