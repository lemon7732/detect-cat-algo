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
- 当前一键脚本会在检测到 `Python` 不在 `3.10-3.12` 范围内时直接终止，并给出明确提示

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
conda create -n catalgo310 python=3.10 -y
conda activate catalgo310
bash scripts/autodl_one_click_train.sh --use-system-python
```

### 方式 B：只训练，不重复装环境

第一次完成后，如果实例没有重置，可以直接复用虚拟环境：

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh --skip-install --skip-download-binary --skip-download-cat-dataset
```

如果你用的是 AutoDL 镜像里的 conda 环境而不是项目自己的虚拟环境，更适合这样跑：

```bash
cd /root/autodl-tmp/algo
bash scripts/autodl_one_click_train.sh \
  --use-system-python \
  --skip-download-binary \
  --skip-download-cat-dataset
```

说明：

- 在 `--use-system-python` 模式下，如果脚本检测到 `nvidia-smi`，会自动安装 `tensorflow[and-cuda]==2.19.0`
- 这比继续复用镜像里自带但不一定匹配的 TensorFlow 更稳
- 脚本也会自动把 `certifi` 的 CA 证书导出给 `TFDS/requests/urllib3`，以缓解 AutoDL 环境里常见的 SSL 下载失败问题
- 如果 AutoDL 证书链依然异常，可以额外使用 `--allow-insecure-ssl-downloads` 作为兜底

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

如果你要手动指定 TensorFlow 安装包：

```bash
bash scripts/autodl_one_click_train.sh \
  --use-system-python \
  --tensorflow-package "tensorflow[and-cuda]==2.19.0"
```

如果你的 TFDS 下载持续报 `CERTIFICATE_VERIFY_FAILED`，可以临时这样运行：

```bash
bash scripts/autodl_one_click_train.sh \
  --use-system-python \
  --allow-insecure-ssl-downloads
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
PYTHONPATH=src python scripts/check_env.py
PYTHONPATH=src python scripts/check_downloads.py
```

### 训练日志

建议把命令输出保存到日志文件：

```bash
bash scripts/autodl_one_click_train.sh 2>&1 | tee artifacts/autodl/train.log
```

### TFDS 下载报 `CERTIFICATE_VERIFY_FAILED`

脚本现在会自动设置：

- `SSL_CERT_FILE`
- `REQUESTS_CA_BUNDLE`
- `CURL_CA_BUNDLE`
- `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`

如果你要手动执行，也可以先运行：

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=$SSL_CERT_FILE
```

如果环境本身证书链就是坏的，仍然失败时，可以临时使用不校验证书的下载模式：

```bash
PYTHONPATH=src python scripts/download_public_datasets.py \
  --datasets cats_vs_dogs \
  --allow-insecure-ssl
```

注意：

- 这只建议用于公共开源数据下载
- 不建议把这种模式用于敏感数据或账号相关请求
