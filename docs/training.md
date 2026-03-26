# 训练说明

## 解释器要求

- 基础工程可在 Python 3.14 下运行部分功能
- `TensorFlow 2.19` 完整训练建议使用 Python 3.10-3.12
- 若当前环境是 Python 3.14，可先运行：

```bash
PYTHONPATH=src python scripts/check_env.py
```

## 二分类模型

```bash
PYTHONPATH=src python scripts/train_binary.py --config configs/binary.yaml
```

公开数据集训练：

```bash
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_cats_vs_dogs.yaml
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_oxford_iiit_pet.yaml
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_thesis_public.yaml
```

输出：
- `artifacts/binary/best.weights.h5`
- `artifacts/binary/final.weights.h5`
- `artifacts/binary/history.csv`
- `artifacts/binary/history.png`
- `artifacts/binary/eval.json`

## 关键点模型

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks.yaml
```

`CatFLW` 训练：

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks_catflw.yaml
```

`CAT Dataset` 原生 9 点训练：

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks_cat_dataset.yaml
```

输出：
- `artifacts/landmarks/best.weights.h5`
- `artifacts/landmarks/final.weights.h5`
- `artifacts/landmarks/history.csv`
- `artifacts/landmarks/history.png`
- `artifacts/landmarks/eval.json`
- `artifacts/landmarks/previews/`

## 说明

- 二分类默认使用论文改进版 `F-ResNet-SE`
- [configs/binary_thesis_public.yaml](/home/lemon/self/code/algo/configs/binary_thesis_public.yaml) 会组合 `cats_vs_dogs`、`Oxford-IIIT Pet`、`CelebA` 和 `Caltech Birds 2011`，更接近论文的 `cat vs not_cat(狗/鸟/人)` 场景
- 关键点模型使用 VGG 风格回归网络
- 若 Haar 级联不可用，关键点预处理会回退到整图模式
- `cats_vs_dogs` 与 `oxford_iiit_pet` 会通过 `tensorflow-datasets` 自动下载
- `CAT Dataset` 是更贴近论文的原生九点训练源
- `CatFLW` 需要本地解压，并在配置里设置 48 点到 9 点的 `point_groups`
