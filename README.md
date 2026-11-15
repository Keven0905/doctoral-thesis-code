# 博士学位论文代码

本仓库按论文第3章、第4章和第5章组织为三个子项目。

| 子项目 | 内容 | 状态 |
|---|---|---|
| chapter3-hyperparameter-optimization | 超参数优化 | 待补充 |
| chapter4-activation-fusion | 激活函数融合 | 待补充 |
| chapter5-model-aggregation | 异构模型聚合与模糊聚合 | 已上传 |

## 第五章运行

使用 Python 3.11，安装 chapter5-model-aggregation/requirements.txt 中的依赖，并在 configs/experiment.json 中设置真实数据路径。

```powershell
Set-Location chapter5-model-aggregation
python -m pip install -r requirements.txt
python main.py all --dataset CMI5
```

可通过以下命令查看单模型、聚合模型、消融实验、独立评估和续训入口。

```powershell
python main.py --help
```
