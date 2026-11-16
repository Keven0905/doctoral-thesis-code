# 第4章 基于多样化激活函数融合优化方法

本项目在第3章 NI-ESOA 超参数寻优的基础上，为 DenseNet 引入递归并联机制激活函数（Recursive Parallel Activation Fusion, RPAF），将 ReLU、Tanh、Leaky ReLU、SiLU 四种具备互补特性的基础激活函数按递进逻辑构成函数嵌套链路，通过多级递归复合实现特征信息重塑。

## 方法概述

### 递归并联机制

与串联机制对各支路输出仅做线性加权融合不同，递归并联机制将前一级激活函数的输出直接作为后一级的输入，形成链式变换链路，突破线性凸组合的表达上限：

- **递归序列**：ReLU → Tanh → Leaky ReLU → SiLU（经敏感性分析与消融实验确定的最优传递序列）
  - 第1级 **ReLU**：对卷积输出原始特征做稀疏筛选，强化边缘、角点、纹理等高显著性正向响应；
  - 第2级 **Tanh**：完成特征分布零中心化校正，缓解深层网络特征均值偏移；
  - 第3级 **Leaky ReLU**（负半轴斜率 0.01）：修复前两级被抑制的小幅负向判别信息，保障递归链路梯度通路；
  - 第4级 **SiLU**：利用自门控全局平滑可微特性，对逐级重塑后的特征做非线性精细化打磨。
- **递推结构**：g₀ = x，g_k = f_k(g_{k-1} + β_k)，各级偏置 β_k ∈ (0,1) 作为网络可学习参数。
- **融合输出**：f(x) = x + Σ w_k·g_k。最外层恒等项 x 构成贯穿全链的残差直连通路，为反向传播提供与递归深度解耦的稳定梯度支撑；即便某一级激活单元落入低梯度区间，残差通路仍保障梯度有效流向输入侧。
- **权重寻优**：融合权重 w₁~w₄ 不随反向传播更新，由 NI-ESOA 全局寻优，满足归一化约束 Σw_k = 1 且 0 ≤ w_k ≤ 1，规避人工固定权重的主观经验偏差。
- **主干集成**：模块嵌入 DenseNet 密集连接块内部替换原有 ReLU，不修改卷积、拼接、批归一化等主干网络拓扑。

### NI-ESOA 联合寻优

在第3章二维超参数寻优的基础上，将搜索空间扩展为六维 [log₁₀(LR), DR, w₁, w₂, w₃, w₄]，实现超参数与激活融合权重的联合优化：

- LR ∈ [1e-6, 1e-4]（对数域采样），DR ∈ [0.1, 0.4]，w_k ∈ [0, 1]（解码时归一化）；
- 适应度函数沿用验证集交叉熵损失最小值；
- 白鹭 A 坐等、B 随机游走、C 环绕包围三策略更新 + 30% 随机接受准则 + 牛顿插值变异。

## 实验设置

| 超参数 | 取值 |
|---|---|
| 批大小 | 32 |
| 最大迭代轮次 | 50 |
| 早停 | 启用（监控验证集损失，patience=10） |
| 寻优维度 | 6（LR、DR、w₁~w₄） |
| 种群数量 | 100 |
| 优化器 | Adam |
| 主干网络 | DenseNet121 + RPAF |

数据处理与评估指标沿用第3章设置（Accuracy、Precision、Recall、F1-Score、AUC）。

## 运行方式

环境依赖：Python、PyTorch、torchvision、scikit-learn、NumPy。

```bash
python -m pip install torch torchvision scikit-learn numpy

# 使用自有数据集（ImageFolder 格式，每个类别一个子文件夹）
python ni_esoa_densenet_rpaf.py --data_root /path/to/dataset

# 快速演示（CIFAR-10 子集，仅用于验证流程）
python ni_esoa_densenet_rpaf.py --demo

# 算力受限时可缩小搜索规模
python ni_esoa_densenet_rpaf.py --data_root /path/to/dataset --pop_size 20 --max_iter 10
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `ni_esoa_densenet_rpaf.py` | 递归并联激活模块、NI-ESOA 六维联合寻优、DenseNet121 集成与训练评估的完整实现 |
