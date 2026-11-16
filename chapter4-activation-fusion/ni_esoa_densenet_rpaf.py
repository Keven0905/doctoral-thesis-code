# -*- coding: utf-8 -*-
"""
NI-ESOA-RPAF-DenseNet: 在 NI-ESOA 调优 DenseNet121 超参数(学习率/Dropout)的基础上,


复现依据:
  第3章: NI-ESOA 对 LR∈[1e-6,1e-4](对数域) 与 DR∈[0.1,0.4] 寻优,
         适应度=验证集交叉熵损失最小值, 早停 patience=10 (表3-4)
  第4章 4.3节: 递归并联机制激活函数 (式4-17~4-24)
    - 递归序列: ReLU -> Tanh -> LeakyReLU -> SiLU (经实验确定的最优传递序列)
    - 递推结构: g0=x, g_k = f_k(g_{k-1} + beta_k), k=1..4
    - 融合输出: f(x) = x + sum_k w_k * g_k,  权重约束 w_k∈[0,1], sum(w)=1 (式4-17/4-24)
    - 恒等项 x 为贯穿全链的残差直连通路, 缓解多级递归的梯度衰减 (式4-22/4-25)
    - 权重 w1~w4 不参与反向传播, 由 NI-ESOA 全局寻优(每一轮迭代后归一化, 式4-13)
    - 模块嵌入 DenseNet 密集连接块内部替换 ReLU, 不改卷积/拼接/BN 主干拓扑

  NI-ESOA 联合寻优空间(6维): [log10(LR), DR, w1, w2, w3, w4]

用法:
  python ni_esoa_densenet_rpaf.py --data_root /path/to/dataset   # ImageFolder 格式
  python ni_esoa_densenet_rpaf.py --demo                          # CIFAR-10 快速演示
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------
# 0. 全局配置(论文表3-4 + 第4章激活函数设置)
# ---------------------------------------------------------------
CFG = dict(
    # ---- 第3章 NI-ESOA 超参数空间 ----
    lr_min=1e-6, lr_max=1e-4,      # LR 搜索边界(表3-4)
    dr_min=0.1,  dr_max=0.4,       # DR 搜索边界
    # ---- 第4章 递归融合权重边界 w_k ∈ [0,1] ----
    w_min=0.0, w_max=1.0,
    # ---- 算法与训练设置 ----
    pop_size=100,                  # 种群规模(论文=100, 可调小以节省算力)
    dim=6,                         # 寻优维度: [log10(lr), dr, w1, w2, w3, w4]
    max_iter=20,                   # NI-ESOA 外层迭代次数
    epochs=50,                     # 每个体训练最大轮次
    patience=10,                   # 早停 patience(监控验证集损失)
    batch_size=32,
    img_size=224,
    accept_prob=0.3,               # 随机接受准则概率(式3-22)
    ni_points=5,                   # 牛顿插值使用的历史最优点数
    leaky_slope=0.01,              # Leaky ReLU 负半轴斜率(论文4.2节取0.01)
    num_recursions=1,              # 递归轮次(式4-17对应1轮; 图4-2步骤5支持多轮回溯)
    seed=42,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# 1. 牛顿插值算子(第3章 式3-1): 差商表递推构造, 支持增量更新
# ---------------------------------------------------------------
class NewtonInterpolator:
    """牛顿差商插值: 对历史最优解序列拟合多项式并外推预测。

    新增数据点只需追加一阶差商计算, 无需重构全部系数,
    契合进化算法迭代中数据点动态增加的场景(论文3.2.2节)。
    """

    def __init__(self, max_points=5):
        self.max_points = max_points
        self.xs = []  # 自变量(迭代序号 t)
        self.ys = []  # 函数值(各维度的全局最优位置分量)

    def add_point(self, x, y):
        self.xs.append(float(x))
        self.ys.append(np.asarray(y, dtype=float))
        if len(self.xs) > self.max_points:
            self.xs.pop(0)
            self.ys.pop(0)

    def _divided_diff(self):
        """构建差商表, 返回各阶差商系数 (n, dim)。"""
        n = len(self.xs)
        coef = np.array(self.ys, dtype=float)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                denom = self.xs[i] - self.xs[i - j]
                denom = denom if abs(denom) > 1e-12 else 1e-12
                coef[i] = (coef[i] - coef[i - 1]) / denom
        return coef

    def predict(self, x_new):
        """嵌套(Horner)形式求值 P_n(x_new)。"""
        n = len(self.xs)
        if n == 0:
            return None
        coef = self._divided_diff()
        result = coef[n - 1].copy()
        for i in range(n - 2, -1, -1):
            result = result * (x_new - self.xs[i]) + coef[i]
        return result

    def ready(self):
        return len(self.xs) >= 2


# ---------------------------------------------------------------
# 2. 第4章: 递归并联机制激活函数模块 RPAF (式4-17~4-24)
# ---------------------------------------------------------------
class RecursiveParallelActivation(nn.Module):
    """递归并联机制复合激活函数。

    递归链路(式4-18~4-21, 递归次序 ReLU->Tanh->LeakyReLU->SiLU):
        g1 = f_ReLU(x  + b1)     稀疏筛选, 强化高显著性正向响应
        g2 = f_Tanh(g1 + b2)     零中心化校正, 动态区间压缩
        g3 = f_LeakyReLU(g2 + b3) 负向弱特征补偿, 保障梯度通路
        g4 = f_SiLU(g3 + b4)     自门控平滑精细化打磨

    融合输出(式4-17/4-24):
        f(x) = x + sum_k w_k * g_k
        x 为恒等残差直连通路(式4-22/4-25 梯度鲁棒性来源);
        w = [w1..w4] 为 NI-ESOA 寻优的跨层级中间特征融合权重,
        满足 sum(w)=1, 0<=w_k<=1 (每轮迭代后归一化, 式4-13);
        beta_k 为各级偏置(论文约束 0~1), 作为网络可学习参数(sigmoid 钳制)。
    """

    def __init__(self, leaky_slope=0.01, num_recursions=1):
        super().__init__()
        self.leaky_slope = leaky_slope
        self.num_recursions = num_recursions
        # 融合权重: 不随反向传播更新, 由 NI-ESOA 从外部写入(buffer 不参与梯度)
        self.register_buffer('fuse_w', torch.full((4,), 0.25))
        # 各级递归单元偏置 beta_k ∈ (0,1), 经 sigmoid 钳制, 随网络训练更新
        self.beta_raw = nn.Parameter(torch.zeros(4))  # sigmoid(0)=0.5 初始化

    def set_fusion_weights(self, w):
        """由 NI-ESOA 写入归一化后的融合权重。"""
        w = torch.as_tensor(w, dtype=self.fuse_w.dtype, device=self.fuse_w.device)
        self.fuse_w.copy_(w / w.sum().clamp_min(1e-12))

    def forward(self, x):
        beta = torch.sigmoid(self.beta_raw)          # 偏置约束在 (0,1)
        out = x
        for _ in range(self.num_recursions):         # 递归特征回溯(图4-2 步骤5)
            g1 = F.relu(out + beta[0])               # 第1级 ReLU
            g2 = torch.tanh(g1 + beta[1])            # 第2级 Tanh
            g3 = F.leaky_relu(g2 + beta[2],
                              negative_slope=self.leaky_slope)  # 第3级 LeakyReLU
            g4 = F.silu(g3 + beta[3])                # 第4级 SiLU
            # 式4-17: 恒等残差 + 各级中间特征按融合权重加权聚合
            out = out + (self.fuse_w[0] * g1 + self.fuse_w[1] * g2
                         + self.fuse_w[2] * g3 + self.fuse_w[3] * g4)
        return out


def build_densenet121_rpaf(num_classes, dropout_rate, fuse_weights,
                           cfg, pretrained=True):
    """DenseNet121 + RPAF 激活 + 可配置 Dropout 分类头。

    将密集连接块/过渡层内的全部 ReLU 替换为递归并联激活模块,
    不修改卷积、拼接、批归一化等主干拓扑(论文4.2/4.3节集成方式)。
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    def replace_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name,
                        RecursiveParallelActivation(cfg['leaky_slope'],
                                                    cfg['num_recursions']))
            else:
                replace_relu(child)

    replace_relu(model.features)

    # 统一写入 NI-ESOA 寻优得到的融合权重(各模块共享同一组权重)
    for m in model.modules():
        if isinstance(m, RecursiveParallelActivation):
            m.set_fusion_weights(fuse_weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------
# 3. 数据管线(沿用第3章 3.3.2节): 224x224, 旋转/裁剪增强, 8:2 划分, 训练集10%作验证
# ---------------------------------------------------------------
def build_dataloaders(data_root, cfg):
    train_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomRotation(degrees=(0, 270)),     # 旋转增强(90/180/270)
        transforms.RandomResizedCrop(cfg['img_size'], scale=(0.6, 0.9)),  # 裁剪60%~90%
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(data_root)
    num_classes = len(full.classes)
    n_total = len(full)
    n_test = int(n_total * 0.2)
    n_trainval = n_total - n_test
    n_val = int(n_trainval * 0.1)
    n_train = n_trainval - n_val

    g = torch.Generator().manual_seed(cfg['seed'])
    trainval_set, test_set = random_split(full, [n_trainval, n_test], generator=g)
    train_set, val_set = random_split(trainval_set, [n_train, n_val], generator=g)

    train_set.dataset = datasets.ImageFolder(data_root, transform=train_tf)
    trainval_eval = datasets.ImageFolder(data_root, transform=eval_tf)
    val_set.dataset = trainval_eval
    test_set.dataset = datasets.ImageFolder(data_root, transform=eval_tf)

    kw = dict(batch_size=cfg['batch_size'], num_workers=2, pin_memory=True)
    loaders = dict(
        train=DataLoader(train_set, shuffle=True, **kw),
        val=DataLoader(val_set, shuffle=False, **kw),
        test=DataLoader(test_set, shuffle=False, **kw),
    )
    return loaders, num_classes


def build_cifar10_dataloaders(cfg):
    """--demo 模式: CIFAR-10 子集快速验证流程。"""
    train_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomResizedCrop(cfg['img_size'], scale=(0.6, 0.9)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    root = './data'
    trainval = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    trainval = torch.utils.data.Subset(trainval, range(2000))
    test = torch.utils.data.Subset(test, range(500))
    n_val = int(len(trainval) * 0.1)
    n_train = len(trainval) - n_val
    g = torch.Generator().manual_seed(cfg['seed'])
    train_set, val_set = random_split(trainval, [n_train, n_val], generator=g)
    kw = dict(batch_size=cfg['batch_size'], num_workers=0)
    loaders = dict(
        train=DataLoader(train_set, shuffle=True, **kw),
        val=DataLoader(val_set, shuffle=False, **kw),
        test=DataLoader(test, shuffle=False, **kw),
    )
    return loaders, 10


# ---------------------------------------------------------------
# 4. 个体位置编解码: [log10(lr), dr, w1, w2, w3, w4]
# ---------------------------------------------------------------
def decode_position(position, cfg):
    """位置向量 -> (lr, dr, 归一化融合权重w)。权重每轮归一化(式4-13)。"""
    lr = 10.0 ** position[0]                          # 对数域解码(式3-9)
    dr = float(position[1])
    w = np.clip(np.asarray(position[2:], dtype=float),
                cfg['w_min'], cfg['w_max'])
    w = w / max(w.sum(), 1e-12)                       # 归一化约束 sum(w)=1
    return lr, dr, w


# ---------------------------------------------------------------
# 5. 适应度函数(式3-11~3-14): 训练后取验证集损失记录最小值
# ---------------------------------------------------------------
def evaluate_fitness(position, loaders, num_classes, cfg, device,
                     epochs=None, verbose=False):
    lr, dr, w = decode_position(position, cfg)
    epochs = epochs or cfg['epochs']

    model = build_densenet121_rpaf(num_classes, dr, w, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []                 # xloss_history(式3-13)
    best_loss, bad_rounds = float('inf'), 0
    for ep in range(epochs):
        model.train()
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(device), y.to(device)
                vloss += criterion(model(x), y).item() * x.size(0)
                n += x.size(0)
        vloss /= max(n, 1)
        loss_history.append(vloss)
        if verbose:
            print(f'    epoch {ep+1:02d}/{epochs}  val_loss={vloss:.4f}')

        if vloss < best_loss - 1e-6:  # 早停 patience=10(表3-4)
            best_loss, bad_rounds = vloss, 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg['patience']:
                break

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return min(loss_history)          # xfit_best = min(xloss_history)(式3-14)


# ---------------------------------------------------------------
# 6. NI-ESOA 算法核心(第3章算法3-1, 扩展至6维联合寻优)
# ---------------------------------------------------------------
class NIESOA:
    """基于牛顿插值的白鹭群优化算法。

    位置向量 x = [log10(LR), DR, w1, w2, w3, w4]:
      前2维对应第3章超参数寻优; 后4维对应第4章递归并联融合权重,
      解码时做归一化约束(式4-13)。每个个体经白鹭A坐等、B随机游走、
      C环绕包围三策略生成候选, 判别条件(更优必接受/否则30%概率, 式3-22)
      筛选后, 再经牛顿插值变异(式3-23)更新。
    """

    def __init__(self, cfg, fitness_fn):
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.lb = np.array([np.log10(cfg['lr_min']), cfg['dr_min'],
                            cfg['w_min'], cfg['w_min'], cfg['w_min'], cfg['w_min']])
        self.ub = np.array([np.log10(cfg['lr_max']), cfg['dr_max'],
                            cfg['w_max'], cfg['w_max'], cfg['w_max'], cfg['w_max']])
        self.hop = self.ub - self.lb
        # 种群初始化: 搜索空间内均匀分布(式3-10/4-13)
        self.pop = self.lb + np.random.rand(cfg['pop_size'], cfg['dim']) * self.hop
        self.fit = np.full(cfg['pop_size'], np.inf)
        self.gbest_pos = None
        self.gbest_fit = np.inf
        self.newton = NewtonInterpolator(max_points=cfg['ni_points'])
        self.curve = []

    def clip(self, x):
        """越界分量截断映射回搜索边界(式3-17)。"""
        return np.clip(x, self.lb, self.ub)

    # ---- 白鹭A: 坐等策略, 伪梯度估计(式3-18/3-19) ----
    def egret_A(self, x, fit_i, t):
        eps = 1e-12
        g_hat = (fit_i - self.gbest_fit) / (np.linalg.norm(x - self.gbest_pos) + eps)
        h = 0.5 * (1 - t / self.cfg['max_iter'])
        direction = self.gbest_pos - x
        return x + h * g_hat * direction / (abs(g_hat) + eps) * np.random.rand()

    # ---- 白鹭B: 激进策略-随机游走(式3-20) ----
    def egret_B(self, x, t):
        sigma = 0.1 * (1 - t / self.cfg['max_iter'])
        return x + sigma * self.hop * np.random.randn(self.cfg['dim'])

    # ---- 白鹭C: 激进策略-环绕包围(式3-21) ----
    def egret_C(self, x):
        return x + np.random.uniform(0, 1) * (self.gbest_pos - x)

    # ---- 牛顿插值变异(式3-23重构) ----
    def newton_mutation(self, x, t):
        """差商表拟合历史最优解序列并外推, 叠加随迭代衰减的正态扰动。"""
        T = self.cfg['max_iter']
        decay = np.exp(-t / T)
        noise = np.random.randn(self.cfg['dim'])
        if self.newton.ready():
            x_pred = self.clip(self.newton.predict(t + 1))
            x_new = x + decay * noise * (x_pred - x)
        else:
            x_new = x + decay * noise * (self.gbest_pos - x)
        return self.clip(x_new)

    # ---- 判别条件(式3-22) ----
    def accept(self, cand_pos, cand_fit, cur_pos, cur_fit):
        if cand_fit < cur_fit:
            return cand_pos, cand_fit
        if np.random.rand() < self.cfg['accept_prob']:
            return cand_pos, cand_fit
        return cur_pos, cur_fit

    # ---- 主循环 ----
    def optimize(self):
        for i in range(self.cfg['pop_size']):
            self.fit[i] = self.fitness_fn(self.pop[i])
            if self.fit[i] < self.gbest_fit:
                self.gbest_fit = self.fit[i]
                self.gbest_pos = self.pop[i].copy()
        self.newton.add_point(0, self.gbest_pos)
        self.curve.append(self.gbest_fit)

        for t in range(1, self.cfg['max_iter'] + 1):
            for i in range(self.cfg['pop_size']):
                x, f = self.pop[i], self.fit[i]
                for cand in (self.egret_A(x, f, t),
                             self.egret_B(x, t),
                             self.egret_C(x)):
                    cand = self.clip(cand)
                    cand_fit = self.fitness_fn(cand)
                    x, f = self.accept(cand, cand_fit, x, f)
                cand = self.newton_mutation(x, t)
                cand_fit = self.fitness_fn(cand)
                x, f = self.accept(cand, cand_fit, x, f)

                self.pop[i], self.fit[i] = x, f
                if f < self.gbest_fit:    # 全局最优更新(式3-15/3-16)
                    self.gbest_fit = f
                    self.gbest_pos = x.copy()

            self.newton.add_point(t, self.gbest_pos)
            self.curve.append(self.gbest_fit)
            lr, dr, w = decode_position(self.gbest_pos, self.cfg)
            print(f'[NI-ESOA] iter {t:02d}/{self.cfg["max_iter"]}  '
                  f'best_val_loss={self.gbest_fit:.4f}  '
                  f'lr={lr:.2e}  dr={dr:.4f}  w={np.round(w, 3).tolist()}')

        return decode_position(self.gbest_pos, self.cfg)


# ---------------------------------------------------------------
# 7. 最终模型训练与评估(沿用第3章指标: Acc/Precision/Recall/F1/AUC)
# ---------------------------------------------------------------
def final_train_and_eval(best_lr, best_dr, best_w, loaders, num_classes,
                         cfg, device):
    print(f'\n使用最优配置训练最终模型: lr={best_lr:.2e}, '
          f'dropout={best_dr:.4f}, w={np.round(best_w, 4).tolist()}')
    model = build_densenet121_rpaf(num_classes, best_dr, best_w, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    best_state, best_loss, bad = None, float('inf'), 0
    for ep in range(cfg['epochs']):
        model.train()
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for x, y in loaders['val']:
                x, y = x.to(device), y.to(device)
                vloss += criterion(model(x), y).item() * x.size(0)
                n += x.size(0)
        vloss /= max(n, 1)
        print(f'  final epoch {ep+1:02d}/{cfg["epochs"]}  val_loss={vloss:.4f}')
        if vloss < best_loss - 1e-6:
            best_loss, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                print('  早停触发')
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loaders['test']:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
            y_prob.append(prob)
            y_pred.append(prob.argmax(1))
            y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    auc = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')

    print('\n===== 测试集评估结果 =====')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-Score : {f1:.4f}')
    print(f'AUC      : {auc:.4f}')
    return model, dict(acc=acc, precision=prec, recall=rec, f1=f1, auc=auc)


# ---------------------------------------------------------------
# 8. 主入口
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='NI-ESOA 优化 DenseNet121(递归并联激活RPAF) 超参数与融合权重')
    parser.add_argument('--data_root', type=str, default='',
                        help='ImageFolder 格式数据集根目录(每类一个子文件夹)')
    parser.add_argument('--demo', action='store_true', help='CIFAR-10 快速演示模式')
    parser.add_argument('--pop_size', type=int, default=CFG['pop_size'])
    parser.add_argument('--max_iter', type=int, default=CFG['max_iter'])
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    args = parser.parse_args()

    cfg = CFG.copy()
    cfg.update(pop_size=args.pop_size, max_iter=args.max_iter, epochs=args.epochs)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    if args.demo or not args.data_root:
        print('演示模式: 使用 CIFAR-10 子集(仅验证流程, 非论文数据集)')
        loaders, num_classes = build_cifar10_dataloaders(cfg)
    else:
        loaders, num_classes = build_dataloaders(args.data_root, cfg)
    print(f'类别数: {num_classes}')

    def fitness(position):
        return evaluate_fitness(position, loaders, num_classes, cfg, device)

    # NI-ESOA 联合寻优 [log10(lr), dr, w1..w4]
    optimizer = NIESOA(cfg, fitness)
    best_lr, best_dr, best_w = optimizer.optimize()
    print(f'\n最优配置: lr={best_lr:.2e}, dropout={best_dr:.4f}, '
          f'w={np.round(best_w, 4).tolist()}, '
          f'最优验证损失={optimizer.gbest_fit:.4f}')

    # 最优配置重训并在测试集评估
    final_train_and_eval(best_lr, best_dr, best_w, loaders, num_classes,
                         cfg, device)


if __name__ == '__main__':
    main()
