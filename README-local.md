# 本地开发环境

环境：Python 3.13，项目独立虚拟环境 `.venv`。

## 打开 Jupyter

双击 `start-jupyter.bat`，会打开 JupyterLab。选择 `Python (makemore)` 内核，然后打开 `bigram.ipynb` 或 `rewrite_bigram.ipynb`。
保留启动窗口；关闭服务可在窗口按 Ctrl+C。

## 命令行

双击 `open-python.bat` 打开已激活环境的终端。运行：

```
python bigram.py
```

也可在 PowerShell 中直接运行：

```
.\.venv\Scripts\python.exe bigram.py
```

## GPU

目前安装 PyTorch 2.10.0 CPU 版（已验证通过本机应用控制策略），可直接运行仓库示例。CUDA 版本安装包下载较慢，未安装 GPU 支持。检查：

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
```

原仓库示例默认使用 CPU。使用 GPU 时需将模型和张量显式放到 `cuda` 设备。

## 路径和依赖

Jupyter 从项目根目录启动，因此 `names.txt` 可以直接读取。
`requirements-lock.txt` 记录安装时的依赖版本。
虚拟环境依赖本机现有的 Python 3.13 安装，请保留该 Python。


