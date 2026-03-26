# 校园流浪猫识别算法工程

本项目实现论文《基于猫脸智能识别的校园流浪猫救助管理系统的设计与实现》中的完整算法链路：

`猫/非猫二分类(F-ResNet-SE) -> Haar 猫脸检测 -> VGG 九点关键点回归 -> 欧氏/余弦 1:N 检索 -> FastAPI 接口`

## 目录结构

- `configs/`: 训练、推理、建库配置
- `src/cat_rescue_ai/`: 核心算法包
- `scripts/`: 命令行入口
- `data/`: 数据目录约定
- `artifacts/`: 模型、日志、可视化与底库索引输出
- `tests/`: 单元测试与 smoke test
- `docs/`: 数据准备、训练、推理、API 文档

## 环境安装

### 基础运行环境

适用于当前仓库的配置加载、OpenCV 检测、API 骨架、测试与通用工具。

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 完整训练环境

`TensorFlow 2.19` 当前建议使用 `Python 3.10-3.12`。如果你现在的解释器是 `Python 3.14`，请在兼容解释器下再安装：

```bash
.venv/bin/pip install -r requirements-ml.txt
```

可以先执行环境检查：

```bash
PYTHONPATH=src python scripts/check_env.py
```

下载公开数据集：

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets cats_vs_dogs oxford_iiit_pet celeb_a caltech_birds2011
```

## 数据目录约定

项目现在同时支持“目录型数据”和“公开数据集源”两种方式：

- 二分类：本地目录、`TFDS cats_vs_dogs`、`TFDS Oxford-IIIT Pet`
- 二分类：也支持混合多个 TFDS 数据源，构建更贴近论文的 `cat vs not_cat(狗/鸟/人)` 训练集
- 关键点：原生 `.cat` 九点标注、本地 `CSV + images`、`CatFLW`
- 个体底库：目录扫描生成 `metadata.csv`，适配校园猫图片或 `CatIndividualImages` 这类按个体分目录的数据

### 二分类数据

```text
data/binary/raw/
  cat/
    xxx.jpg
  not_cat/
    yyy.jpg
```

### 关键点数据

```text
data/cat_dataset/
  CAT_00/
    00000001_000.jpg
    00000001_000.jpg.cat

data/landmarks/
  images/
    cat_001.jpg
  train.csv
```

`train.csv` 默认字段支持：
- `image_id`
- `left_eye_x`, `left_eye_y`
- `right_eye_x`, `right_eye_y`
- `mouth_x`, `mouth_y`
- `left_ear1_x`, `left_ear1_y`
- `left_ear2_x`, `left_ear2_y`
- `left_ear3_x`, `left_ear3_y`
- `right_ear1_x`, `right_ear1_y`
- `right_ear2_x`, `right_ear2_y`
- `right_ear3_x`, `right_ear3_y`

### 校园猫底库

```text
data/gallery/
  metadata.csv
  cat_001/
    1.jpg
    2.jpg
  cat_002/
    1.jpg
```

`metadata.csv` 最小字段：
- `cat_id`
- `name`
- `sex`
- `age`
- `description`
- `image_path`

## 常用命令

### 训练二分类模型

```bash
PYTHONPATH=src python scripts/train_binary.py --config configs/binary.yaml
```

使用公开数据集：

```bash
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_cats_vs_dogs.yaml
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_oxford_iiit_pet.yaml
PYTHONPATH=src python scripts/train_binary.py --config configs/binary_thesis_public.yaml
```

### 评估二分类模型

```bash
PYTHONPATH=src python scripts/eval_binary.py --config configs/binary.yaml
```

### 训练关键点模型

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks.yaml
```

使用 `CatFLW`：

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks_catflw.yaml
```

使用 `CAT Dataset` 原生九点标注：

```bash
PYTHONPATH=src python scripts/train_landmarks.py --config configs/landmarks_cat_dataset.yaml
```

### 建立猫底库

```bash
PYTHONPATH=src python scripts/build_gallery.py --config configs/gallery.yaml
```

如果你的图库是“每只猫一个文件夹”，可以先自动生成元数据：

```bash
PYTHONPATH=src python scripts/generate_gallery_metadata.py --gallery-root data/gallery --output data/gallery/metadata.csv
```

### 单图识别

```bash
PYTHONPATH=src python scripts/predict_image.py --config configs/api.yaml --image /path/to/image.jpg
```

### 启动 API

```bash
PYTHONPATH=src python scripts/serve_api.py --config configs/api.yaml
```

### 运行内置 Demo

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-demo .venv312/bin/python scripts/run_demo_pipeline.py
```

### AutoDL 一键训练

```bash
bash scripts/autodl_one_click_train.sh
```

详细说明见 [docs/autodl.md](/home/lemon/self/code/algo/docs/autodl.md)。

## 说明

- 二分类模型支持 `resnet50_transfer` 与论文改进版 `f_resnet_se`
- 猫脸检测默认使用 `haarcascade_frontalcatface.xml`
- 系统建库默认采用模型自动预测关键点，而不是人工标注关键点
- 当前仓库不包含公开数据集与训练后权重，需按文档自行准备
- `cats_vs_dogs`、`Oxford-IIIT Pet` 通过 `tensorflow-datasets` 自动下载到本地缓存
- 可用 [download_public_datasets.py](/home/lemon/self/code/algo/scripts/download_public_datasets.py) 统一下载公开数据
- 二分类更贴近论文的公开数据配置见 [binary_thesis_public.yaml](/home/lemon/self/code/algo/configs/binary_thesis_public.yaml)
- 关键点训练更推荐使用 `crawford/cat-dataset` 这类原生 9 点 CAT Dataset
- `CatFLW` 需要你先下载并解压到本地，再按 [docs/data_preparation.md](/home/lemon/self/code/algo/docs/data_preparation.md) 配置
- 在 Python 3.14 下可以先跑基础层和 OpenCV 检测；完整 TensorFlow 训练建议切换到 Python 3.10-3.12
- 仓库提供了合成 demo 数据链路，见 [docs/demo.md](/home/lemon/self/code/algo/docs/demo.md)
- AutoDL 上云训练流程见 [docs/autodl.md](/home/lemon/self/code/algo/docs/autodl.md)
