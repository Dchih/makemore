#!/bin/zsh
set -eu

# Finder may start the script from a different working directory.
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"

fail() {
  print -u2 -- "$1"
  if [[ -t 0 ]]; then
    read -r "?按回车键关闭窗口……"
  fi
  exit 1
}

[[ -x "$PYTHON" ]] || fail "找不到项目的 Python 环境：$PYTHON"
if ! "$PYTHON" -c 'import jupyterlab' >/dev/null 2>&1; then
  fail "项目环境尚未安装 JupyterLab。请进入项目目录执行 uv pip install jupyterlab。"
fi

# Keep local configuration and caches alongside the Windows launcher.
export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export IPYTHONDIR="$PROJECT_DIR/.jupyter/ipython"
export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.jupyter"
export JUPYTER_DATA_DIR="$PROJECT_DIR/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.jupyter/runtime"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

print -- "正在启动 makemore 的 JupyterLab……"
print -- "项目目录：$PROJECT_DIR"
print -- "首选地址：http://127.0.0.1:8889/lab（端口被占用时自动更换）"
print -- "浏览器会自动打开；实际地址以接下来的 Jupyter 输出为准。"
print -- "请保留本窗口。停止服务时按 Ctrl+C，并按提示确认。"
print

exec "$PYTHON" -m jupyterlab \
  --ip=127.0.0.1 \
  --port=8889 \
  --port-retries=10 \
  --notebook-dir="$PROJECT_DIR" \
  --ServerApp.open_browser=True \
  "$@"
