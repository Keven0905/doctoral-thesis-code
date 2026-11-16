# -*- coding: utf-8 -*-
"""
NI-ESOA-DenseNet: 基于牛顿插值的白鹭群优化算法(NI-ESOA)对 DenseNet121 的
学习率(LR)与 Dropout 率(DR)进行联合超参数寻优的图像分类代码。

  1. 超参数空间: LR ∈ [1e-6, 1e-4](对数域均匀采样), DR ∈ [0.1, 0.4]  (式3-8/3-9/3-10)
  2. 适应度函数: 配置指导模型训练后, 取验证集交叉熵损失记录的最小值 (式3-11~3-16)
  3. 边界处理: 越界分量截断映射回搜索边界 (式3-17)
  4. 种群更新: 白鹭A坐等策略(伪梯度估计, 式3-18/3-19)、白鹭B随机游走(式3-20)、
     白鹭C环绕包围(式3-21)、30%概率随机接受准则(式3-22)
  5. 牛顿插值改进: 差商表递推构造插值多项式, 拟合历史最优解序列并施加
     NI 变异扰动 (式3-23, 论文公式为图片, 按文字描述忠实重构)

用法:
  python ni_esoa_densenet.py --data_root /path/to/dataset   # ImageFolder 格式
  python ni_esoa_densenet.py --demo                          # CIFAR-10 快速演示
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------
# 0. 全局配置(论文表3-4)
# ---------------------------------------------------------------
CFG = dict(
    lr_min=1e-6, lr_max=1e-4,      # LR 搜索边界(论文表3-4)
    dr_min=0.1,  dr_max=0.4,       # DR 搜索边界
    pop_size=100,                  # 种群规模(论文=100, 可调小以节省算力)
    dim=2,                         # 超参数维度 (LR, DR)
    max_iter=20,                   # NI-ESOA 外层迭代次数(论文未明示, 可配置)
    epochs=50,                     # 每个体训练最大轮次
    patience=10,                   # 早停 patience(监控验证集损失)
    batch_size=32,
    img_size=224,
    accept_prob=0.3,               # 随机接受准则概率(式3-22)
    ni_points=5,                   # 牛顿插值使用的历史最优点数
    seed=42,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# 1. 牛顿插值算子(式3-1): 差商表递推构造, 支持增量更新
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
        """新增数据点 (x, y向量), 超出容量时丢弃最旧点。"""
        self.xs.append(float(x))
        self.ys.append(np.asarray(y, dtype=float))
        if len(self.xs) > self.max_points:
            self.xs.pop(0)
            self.ys.pop(0)

    def _divided_diff(self):
        """构建差商表, 返回各阶差商系数 (n+1, dim)。"""
        n = len(self.xs)
        coef = np.array(self.ys, dtype=float)          # 零阶差商
        for j in range(1, n):                          # 递推构造 j 阶差商
            for i in range(n - 1, j - 1, -1):
                denom = self.xs[i] - self.xs[i - j]
                denom = denom if abs(denom) > 1e-12 else 1e-12
                coef[i] = (coef[i] - coef[i - 1]) / denom
        return coef

    def predict(self, x_new):
        """嵌套(Horner)形式求值 P_n(x_new), 返回 dim 维预测向量。"""
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
# 2. DenseNet121 模型构建
# ---------------------------------------------------------------
def build_densenet121(num_classes, dropout_rate, pretrained=True):
    """DenseNet121 + 可配置 Dropout 的分类头。"""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    # 分类头加入 Dropout, DR 为待优化超参数
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------
# 3. 数据管线(论文3.3.2节): 224x224, 旋转/裁剪增强, 8:2 划分, 训练集10%作验证
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

    # 训练子集应用增强变换, 验证/测试用确定性变换
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
    """--demo 模式: 使用 CIFAR-10 子集快速验证流程可运行。"""
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
    # 子采样加速演示
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
# 4. 适应度函数(式3-11~3-14): 训练后取验证集损失记录最小值
# ---------------------------------------------------------------
def evaluate_fitness(position, loaders, num_classes, cfg, device,
                     epochs=None, verbose=False):
    """position = [log10(lr), dr]; 返回该配置的最小验证集交叉熵损失。"""
    lr = 10.0 ** position[0]          # 对数域解码(式3-9)
    dr = float(position[1])
    epochs = epochs or cfg['epochs']

    model = build_densenet121(num_classes, dr).to(device)
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

        # 验证集损失
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

        # 早停: 监控验证集损失, patience=10(表3-4)
        if vloss < best_loss - 1e-6:
            best_loss, bad_rounds = vloss, 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg['patience']:
                break

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return min(loss_history)            # xfit_best = min(xloss_history)(式3-14)


# ---------------------------------------------------------------
# 5. NI-ESOA 算法核心
# ---------------------------------------------------------------
class NIESOA:
    """基于牛顿插值的白鹭群优化算法(算法3-1复现)。

    位置向量 x = [log10(LR), DR], 在解码时 LR=10^x0 (对数域采样, 式3-9)。
    每个个体模拟白鹭小队的三种策略(A坐等/B游走/C环绕)生成候选位置,
    经判别条件(更优则接受, 否则30%概率接受)与牛顿插值变异得到新位置。
    """

    def __init__(self, cfg, fitness_fn):
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.lb = np.array([np.log10(cfg['lr_min']), cfg['dr_min']])
        self.ub = np.array([np.log10(cfg['lr_max']), cfg['dr_max']])
        self.hop = self.ub - self.lb
        # 种群初始化: 搜索空间内均匀分布(式3-10)
        self.pop = self.lb + np.random.rand(cfg['pop_size'], cfg['dim']) * self.hop
        self.fit = np.full(cfg['pop_size'], np.inf)
        self.gbest_pos = None          # x_best 全局最优位置
        self.gbest_fit = np.inf        # xfit_best_all(式3-16)
        self.newton = NewtonInterpolator(max_points=cfg['ni_points'])
        self.curve = []                # 收敛曲线记录

    # ---- 边界截断映射(式3-17) ----
    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    # ---- 白鹭A: 坐等策略, 伪梯度估计(式3-18/3-19) ----
    def egret_A(self, x, fit_i, t):
        eps = 1e-12
        # 伪梯度: 由当前位置与全局最优位置的适应度差异构造
        g_hat = (fit_i - self.gbest_fit) / (np.linalg.norm(x - self.gbest_pos) + eps)
        h = 0.5 * (1 - t / self.cfg['max_iter'])     # 步长系数随迭代衰减
        direction = self.gbest_pos - x
        return x + h * g_hat * direction / (abs(g_hat) + eps) * np.random.rand()

    # ---- 白鹭B: 激进策略-随机游走(式3-20), 各向同性高斯扰动 ----
    def egret_B(self, x, t):
        sigma = 0.1 * (1 - t / self.cfg['max_iter'])
        xi = np.random.randn(self.cfg['dim'])
        return x + sigma * self.hop * xi

    # ---- 白鹭C: 激进策略-环绕包围(式3-21), 向全局最优收缩 ----
    def egret_C(self, x):
        q = np.random.uniform(0, 1)
        return x + q * (self.gbest_pos - x)

    # ---- 牛顿插值变异(式3-23重构) ----
    def newton_mutation(self, x, t):
        """用差商表拟合历史最优解序列, 外推预测下一潜在最优区域,
        并叠加随迭代衰减的正态扰动, 增强跳出局部极值的能力。"""
        T = self.cfg['max_iter']
        decay = np.exp(-t / T)                       # exp(-t/tmax) 衰减因子
        noise = np.random.randn(self.cfg['dim'])     # 正态分布随机数
        if self.newton.ready():
            x_pred = self.clip(self.newton.predict(t + 1))  # 插值外推预测
            x_new = x + decay * noise * (x_pred - x)
        else:                                        # 历史点不足时退化为向gbest扰动
            x_new = x + decay * noise * (self.gbest_pos - x)
        return self.clip(x_new)

    # ---- 判别条件(式3-22): 更优必接受, 否则30%概率接受 ----
    def accept(self, cand_pos, cand_fit, cur_pos, cur_fit):
        if cand_fit < cur_fit:
            return cand_pos, cand_fit
        if np.random.rand() < self.cfg['accept_prob']:
            return cand_pos, cand_fit
        return cur_pos, cur_fit

    # ---- 主循环 ----
    def optimize(self):
        # 初始种群适应度评估
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
                # 小队三策略生成候选位置, 逐个经判别条件筛选
                for cand in (self.egret_A(x, f, t),
                             self.egret_B(x, t),
                             self.egret_C(x)):
                    cand = self.clip(cand)
                    cand_fit = self.fitness_fn(cand)
                    x, f = self.accept(cand, cand_fit, x, f)
                # 牛顿插值变异(式3-23)
                cand = self.newton_mutation(x, t)
                cand_fit = self.fitness_fn(cand)
                x, f = self.accept(cand, cand_fit, x, f)

                self.pop[i], self.fit[i] = x, f
                # 全局最优更新(式3-15/3-16)
                if f < self.gbest_fit:
                    self.gbest_fit = f
                    self.gbest_pos = x.copy()

            self.newton.add_point(t, self.gbest_pos)  # 记录历史最优供差商拟合
            self.curve.append(self.gbest_fit)
            print(f'[NI-ESOA] iter {t:02d}/{self.cfg["max_iter"]}  '
                  f'best_val_loss={self.gbest_fit:.4f}  '
                  f'lr={10**self.gbest_pos[0]:.2e}  dr={self.gbest_pos[1]:.4f}')

        best_lr = 10.0 ** self.gbest_pos[0]
        best_dr = float(self.gbest_pos[1])
        return best_lr, best_dr


# ---------------------------------------------------------------
# 6. 最终模型训练与评估(Accuracy/Precision/Recall/F1/AUC)
# ---------------------------------------------------------------
def final_train_and_eval(best_lr, best_dr, loaders, num_classes, cfg, device):
    print(f'\n使用最优超参数训练最终模型: lr={best_lr:.2e}, dropout={best_dr:.4f}')
    model = build_densenet121(num_classes, best_dr).to(device)
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

    # 测试集评估
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
# 7. 主入口
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='NI-ESOA 优化 DenseNet121 超参数')
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

    # 适应度函数闭包
    def fitness(position):
        return evaluate_fitness(position, loaders, num_classes, cfg, device)

    # NI-ESOA 寻优
    optimizer = NIESOA(cfg, fitness)
    best_lr, best_dr = optimizer.optimize()
    print(f'\n最优超参数: lr={best_lr:.2e}, dropout={best_dr:.4f}, '
          f'最优验证损失={optimizer.gbest_fit:.4f}')

    # 最优配置重训并在测试集评估
    final_train_and_eval(best_lr, best_dr, loaders, num_classes, cfg, device)


if __name__ == '__main__':
    main()
