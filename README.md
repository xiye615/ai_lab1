# 实验一：新闻文本十分类

## 环境与数据

- Python 3.10.11；依赖版本见 `requirements.txt`。
- `train_data.csv`：7,368 篇有标签文本；`test_data_unlabeled.csv`：2,457 篇无标签文本。
- `draft_main.py` 在有标签数据上以 `random_state=42` 分层划分 80% 训练集、20% 验证集。
- 保留主题行（Subject）和正文，去除其余邮件元信息、引用、签名、网址、邮箱和标点；将 `c++`、`c#` 转为可保留的词。
- 使用 20,000 维 TF-IDF；保留 `no`、`not`、`nor`、`never` 等否定词。向量器仅在训练子集上拟合。

## 算法与候选参数

所有模型共用同一组 20,000 维 TF-IDF 特征和相同验证集。比较指标为验证集 `macro_f1` 和 `accuracy`，优先按 `macro_f1` 选取配置。

| 算法 | 候选参数 |
| --- | --- |
| MultinomialNB | `alpha=0.1, 1, 10` |
| SVC 线性核 | `C=0.1, 1, 10` |
| SVC RBF 核 | `(C=1, gamma=scale)`、`(C=1, gamma=0.1)`、`(C=10, gamma=scale)` |
| LogisticRegression | `C=0.1, 1, 10, 100`；`solver=lbfgs`、`max_iter=2000` |
| MLPClassifier | `(50,), alpha=0.0001`；`(100,), alpha=0.0001/0.001`；`(100, 50), alpha=0.001`；`max_iter=300`、提前停止 |

线性 SVM 是高维稀疏文本的优先候选；RBF 组合用于比较核函数与 `gamma`。逻辑回归固定求解器，仅比较正则化强度。MLP 比较小型单隐层与两隐层结构。

## 运行

```powershell
python -m pip install -r requirements.txt
python draft_main.py
```

默认命令只显示帮助，**不会训练分类器**。查看全部候选配置：

```powershell
python draft_main.py --list
```

检查代码后，可先训练单一种类，例如 SVM 或逻辑回归：

```powershell
python draft_main.py --model svm --train
python draft_main.py --model logistic_regression --train
```

也可依次训练全部四种算法：

```powershell
python draft_main.py --train
```

这会依次训练 17 组配置。每完成一组都会更新 `results/validation_results.csv`（全部已完成配置）及 `best_parameters.csv`（每种算法当前最佳参数、验证分数和训练耗时）；全部完成后再生成 `validation_comparison.png` 参数对比图。已有完整验证结果时，生成无标签测试集预测只需运行：

```powershell
python draft_main.py --predict
```

该命令读取已保存的最佳参数，不再重复训练其他候选配置；程序在全部有标签数据上拟合 TF-IDF 和该模型，再将预测写入单列、无表头的 `predictions.csv`（覆盖已有同名文件）。如果使用 `--model svm` 等单一算法选项，输出为 `predictions_svm.csv` 等独立文件。这里的“最佳”指验证集 `macro_f1` 最高、`accuracy` 次之的配置；无标签测试集不参与调参或拟合。
