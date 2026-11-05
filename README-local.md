# 本地开发环境

## macOS

双击 `start-jupyter.command`，会使用项目的 `.venv` 环境启动 JupyterLab，并自动打开浏览器。脚本所在目录就是 Notebook 的文件根目录。

首选地址为 `http://127.0.0.1:8889/lab`；端口被占用时会自动更换，实际地址以启动窗口输出为准。保留启动窗口；关闭服务可在窗口按 Ctrl+C，并按提示确认。

## Windows

环境：Python 3.13，项目独立虚拟环境 `.venv`。

### 打开 Jupyter

双击 `start-jupyter.bat`，会打开 JupyterLab。选择 `Python (makemore)` 内核，然后打开 `bigram.ipynb` 或 `rewrite_bigram.ipynb`。
保留启动窗口；关闭服务可在窗口按 Ctrl+C。

### 命令行

双击 `open-python.bat` 打开已激活环境的终端。运行：

```
python bigram.py
```

也可在 PowerShell 中直接运行：

```
.\.venv\Scripts\python.exe bigram.py
```

### GPU

Windows 可以分别使用两个独立环境：

- `.venv`：原有 PyTorch 2.10.0 CPU 环境。
- `.venv-cuda`：PyTorch 2.10.0 CUDA 12.6 环境，用于 RTX 4080 SUPER。

双击 `start-jupyter.bat`，打开 `MLP-cuda.ipynb`，选择 **Python (makemore CUDA)** 内核。第一个代码单元格会显示实际使用的设备和显卡名称。该笔记本保留 2 维 embedding、300 个隐藏单元、batch size 32，以及前 30,000 步学习率 0.1、后 10,000 步学习率 0.01 的实验。

`MLP-cuda.ipynb` 会把参数、数据和 batch 索引放在同一设备上；loss 留在设备上记录，训练结束后统一转回 CPU；计时前后同步 CUDA，显示实际训练耗时。原有 `MLP.ipynb` 可继续单独使用。

当前模型很小，GPU 的启动开销可能高于计算耗时。使用 CPU 时，在导入 `torch` 后加入以下代码，再训练并比较速度：

```python
torch.set_num_threads(1)
```

本机在上述模型与 40,000 步学习率安排下，训练循环实测如下（不含数据构造、评估及内核启动）：

| 计算设备 | 训练耗时 |
| --- | --- |
| CPU，1 个线程 | 6.69 秒 |
| RTX 4080 SUPER | 25.33 秒 |

当前练习优先使用单线程 CPU。更大的模型或 batch 可以重新实测，再决定是否切换 GPU。

在 `MLP-cuda.ipynb` 中也可把 `device` 改为 `torch.device("cpu")` 来对比。显卡环境检查：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # GPU 环境应为 True
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

需要重建 CUDA 环境时，在项目根目录运行以下 PowerShell 命令（需已安装 `uv`，官方 CUDA 安装包约 2.6 GB）：

```powershell
.\.venv\Scripts\python.exe -m venv --without-pip .venv-cuda
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache\uv'
uv pip install --python .\.venv-cuda\Scripts\python.exe 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .\.venv-cuda\Scripts\python.exe --constraint requirements-lock.txt ipykernel matplotlib numpy
.\.venv-cuda\Scripts\python.exe -m ipykernel install --prefix .venv --name makemore-cuda --display-name "Python (makemore CUDA)"
```

安装源及版本对应关系见 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v2100)。`requirements-lock.txt` 记录原 CPU 环境；创建 CUDA 环境时按上面的命令分开安装。

### 路径和依赖

Jupyter 从项目根目录启动，因此 `names.txt` 可以直接读取。
`requirements-lock.txt` 记录安装时的依赖版本。
虚拟环境依赖本机现有的 Python 3.13 安装，请保留该 Python。


