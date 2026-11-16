# 第3章 基于牛顿插值的白鹭群超参数优化方法

针对 DenseNet 图像分类模型中学习率（Learning Rate, LR）与 Dropout 率（Dropout Rate, DR）对分类性能的关键影响，本项目采用基于牛顿插值的白鹭群优化算法（Newton Interpolation-based Egret Swarm Optimization Algorithm, NI-ESOA）对两项超参数进行联合自动寻优。

## 方法概述

- **超参数空间**：将 LR 与 DR 编码为白鹭种群个体的二维位置向量。LR 在 [1e-6, 1e-4] 区间内按对数域均匀采样，DR 在 [0.1, 0.4] 区间内采样，保证寻优过程覆盖学习率的多个数量级。
- **适应度函数**：每组候选超参数指导模型完成训练，取验证集交叉熵损失记录中的最小值作为该个体的适应度值，引导种群向使损失最小化的超参数组合收敛。
- **白鹭群三策略位置更新**：白鹭 A 执行坐等策略，基于伪梯度估计完成局部精细开发；白鹭 B 采用随机游走、白鹭 C 采用环绕包围策略完成全局勘探；经判别条件筛选（更优必接受，否则以 30% 概率随机接受）更新种群位置。
- **牛顿插值改进**：基于差商表递推构造牛顿插值多项式，利用其新增数据点仅需追加一阶差商计算、无需重构全部系数的特性，对历史最优解序列进行拟合并外推预测潜在最优区域，叠加随迭代轮次衰减的正态扰动，增强算法跳出局部极值的能力。
- **边界处理**：个体位置越界分量经截断映射回搜索边界，保障种群持续在有效空间内搜索。

## 实验设置

| 超参数 | 取值 |
|---|---|
| 批大小 | 32 |
| 最大迭代轮次 | 50 |
| 早停 | 启用（监控验证集损失，patience=10） |
| LR 搜索区间 | [1e-6, 1e-4]（对数域） |
| DR 搜索区间 | [0.1, 0.4] |
| 种群数量 | 100 |
| 寻优维度 | 2（LR、DR） |
| 优化器 | Adam |
| 主干网络 | DenseNet121 |

数据处理：图像统一缩放至 224×224，训练集采用旋转（90°/180°/270°）与 60%~90% 随机裁剪增强；数据集按 8:2 划分训练与测试，训练集中再取 10% 作为验证集。

评估指标：Accuracy、Precision、Recall、F1-Score、AUC。

## 运行方式

环境依赖：Python、PyTorch、torchvision、scikit-learn、NumPy。

```bash
python -m pip install torch torchvision scikit-learn numpy

# 使用自有数据集（ImageFolder 格式，每个类别一个子文件夹）
python ni_esoa_densenet.py --data_root /path/to/dataset

# 快速演示（CIFAR-10 子集，仅用于验证流程）
python ni_esoa_densenet.py --demo

# 算力受限时可缩小搜索规模
python ni_esoa_densenet.py --data_root /path/to/dataset --pop_size 20 --max_iter 10
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `ni_esoa_densenet.py` | NI-ESOA 算法、DenseNet121 模型构建、数据管线与训练评估的完整实现 |
