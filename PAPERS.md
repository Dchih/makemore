# makemore 相关论文记录

按项目学习进度整理已用到的论文，后续推进到新阶段时持续追加。

## 当前阶段论文

### 1. A Neural Probabilistic Language Model（Part 2：MLP）

- **作者 / 年份**：Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin (JMLR, 2003)
- **链接**：https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- **核心思想**：用可学习的低维稠密向量（Embedding / Lookup Table `C`）表示离散符号，将上下文向量拼接后经 `tanh` 隐藏层和 Softmax 输出层预测下一个符号，缓解传统 N-gram 的维度灾难与泛化问题。
- **项目对应位置**：`MLP.ipynb`、`MLP.py`、`MLP-cuda.ipynb`（字符查找表 `C`、`block_size = 3` 上下文拼接、`tanh(X @ W1 + B1)`、输出层 `W2/B2`、`F.cross_entropy`）。

### 2. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification（Part 3：初始化）

- **作者 / 年份**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (ICCV, 2015)
- **链接**：https://arxiv.org/abs/1502.01852
- **核心思想**：提出 **Kaiming 初始化（He Initialization）**，分析前向激活与反向梯度经过线性层与非线性激活函数时的方差变化，提出按 `gain / sqrt(fan_in)` 缩放初始权重以稳定信号与梯度尺度。PyTorch 将其推广到各激活函数，对 `tanh` 推荐 `gain = 5/3`。
- **项目对应位置**：`MLP_p3.ipynb`（Cells 11–16，`x @ w` 标准差验证、`fan_in` 缩放实验、`kaiming_scale = (5/3)/30**0.5` 与 `tanh` 饱和 / 死神经元实验）。
- **延伸参考**：Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*（Xavier 初始化，专门针对 `tanh`/sigmoid 网络的早期初始化分析）。

### 3. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（Part 3：BatchNorm）

- **作者 / 年份**：Sergey Ioffe, Christian Szegedy (ICML, 2015)
- **链接**：https://arxiv.org/abs/1502.03167
- **核心思想**：
  - 在隐藏层激活函数之前，按当前 mini-batch 的样本维度对每个特征通道计算均值和方差，将预激活值标准化为零均值、单位方差，缓解深层网络训练时的分布漂移与激活饱和问题，降低对权重初始化尺度和学习率的敏感度。
  - 引入可学习的缩放参数 $\gamma$（gain）和平移参数 $\beta$（bias），保留网络表达能力（不再强行把分布限制在标准正态）。
  - 推理/评估阶段（尤其是单样本生成时）无法依赖当前 batch 统计量，因此需要用训练期间累计的运行均值与方差（running mean / running variance，通常用指数滑动平均更新）。
- **项目对应位置**：`MLP_p3.ipynb` / `LEARNING_PLAN.md` 第 1 阶段（BatchNorm 前向标准化、可训练 gain/bias，以及训练与评估/生成状态区分）。
