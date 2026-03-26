# AutoDL 训练指南

本项目已经提供 AutoDL 一键训练脚本：

- [autodl_one_click_train.sh](/home/lemon/self/code/algo/scripts/autodl_one_click_train.sh)

默认会完成：

1. 创建虚拟环境
2. 安装项目依赖
3. 下载二分类公开数据集
4. 下载 `CAT Dataset`
5. 执行环境检查和数据检查
6. 训练论文风格二分类模型
7. 训练九点关键点模型

## 推荐实例

建议优先选择：

- 单卡 `RTX 4090 24GB`
- 备选 `A5000 24GB`

如果你的 AutoDL 镜像页面只能选 TensorFlow 镜像，且只提供：

- `TensorFlow 2.9.0`
- `TensorFlow 2.5.0`
- `TensorFlow 1.15.5`

那么应该选：

- `TensorFlow 2.9.0`

原因：

- 它是三者里最新的版本
- 本项目使用的是 `tf.keras` 训练链路，`TensorFlow 1.15.5` 不适合
- `TensorFlow 2.5.0` 过旧，兼容性风险明显高于 `2.9.0`

注意：

- 当前仓库主分支代码是按 `Python 3.10+` 风格编写和验证的
- 如果这个 TensorFlow 2.9 镜像固定为 `Python 3.8`，那么镜像虽然能创建，但当前代码还不能保证无改动直接运行
- 也就是说，镜像选择上应当选 `TensorFlow 2.9.0`，但代码层面仍需要继续做 `Python 3.8 / TF 2.9` 兼容改造

## 前置准备

### 1. 创建实例

创建实例后，打开 `JupyterLab` 或通过 `SSH` 登录。

### 2. 准备 Kaggle 凭证

为了下载 `crawford/cat-dataset`，你需要提前准备：

```text
~/.kaggle/kaggle.json
```

放置命令：

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
```

然后把 `kaggle.json` 上传到 `~/.kaggle/kaggle.json`。

## 推荐执行方式

### 方式 A：JupyterLab 终端

在 JupyterLab 里打开终端后执行：

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> algo
cd algo
bash scripts/autodl_one_click_train.sh --use-system-python --use-preinstalled-tensorflow
```

### 方式 B：只训练，不重复装环境

第一次完成后，如果实例没有重置，可以直接复用虚拟环境：

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh --skip-install --skip-download-binary --skip-download-cat-dataset
```

如果你用的是 AutoDL 自带 TensorFlow 镜像而不是项目自己的虚拟环境，更适合这样跑：

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh \
  --use-system-python \
  --use-preinstalled-tensorflow \
  --skip-download-binary \
  --skip-download-cat-dataset
```

### 方式 C：只训练二分类

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh --skip-install --skip-download-cat-dataset --skip-landmarks-train
```

### 方式 D：只训练关键点

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh --skip-install --skip-download-binary --skip-binary-train
```

## 默认训练配置

脚本默认使用：

- 二分类：[binary_thesis_public.yaml](/home/lemon/self/code/algo/configs/binary_thesis_public.yaml)
- 关键点：[landmarks_cat_dataset.yaml](/home/lemon/self/code/algo/configs/landmarks_cat_dataset.yaml)

如果你要改配置：

```bash
bash scripts/autodl_one_click_train.sh \
  --binary-config configs/binary_cats_vs_dogs.yaml \
  --landmarks-config configs/landmarks_cat_dataset.yaml
```

## 输出位置

二分类输出：

- `artifacts/binary_thesis_public/`

关键点输出：

- `artifacts/landmarks_cat_dataset/`

## 常见问题

### 只能选 TensorFlow 2.9 / Python 3.8 镜像

镜像层面的最佳选择是 `TensorFlow 2.9.0`。

但要注意：

- 当前仓库代码还不是原生 `Python 3.8` 兼容版本
- 所以如果你严格只能使用这个镜像，那么下一步应该先做代码兼容改造，再启动正式训练

换句话说：

- 镜像选择：`TensorFlow 2.9.0`
- 代码现状：还需要继续适配 `Python 3.8`

### 没有 Kaggle 凭证

如果你暂时没有 `kaggle.json`，可以先跳过 `CAT Dataset` 下载：

```bash
bash scripts/autodl_one_click_train.sh --skip-download-cat-dataset --skip-landmarks-train
```

### 只想检查环境

```bash
PYTHONPATH=src .venv-autodl/bin/python scripts/check_env.py
PYTHONPATH=src .venv-autodl/bin/python scripts/check_downloads.py
```

### 训练日志

建议把命令输出保存到日志文件：

```bash
bash scripts/autodl_one_click_train.sh 2>&1 | tee artifacts/autodl/train.log
```
