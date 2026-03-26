# Demo 流程

仓库内置了一套可自动生成的演示数据，可以在没有真实数据集时先验证完整算法链路。

## 1. 生成 demo 数据

```bash
PYTHONPATH=src .venv312/bin/python scripts/bootstrap_demo_data.py
```

输出目录：

- `data/demo/binary/raw`
- `data/demo/landmarks/images`
- `data/demo/gallery`
- `data/demo/query`

## 2. 一键跑 demo 训练与识别

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl-demo .venv312/bin/python scripts/run_demo_pipeline.py
```

该脚本会依次执行：

1. 生成 demo 数据
2. 训练 demo 二分类模型
3. 训练 demo 关键点模型
4. 构建 demo 猫底库
5. 对 `data/demo/query` 中的图片做识别

## 3. 单独运行 demo 配置

```bash
PYTHONPATH=src .venv312/bin/python scripts/train_binary.py --config configs/demo_binary.yaml
PYTHONPATH=src .venv312/bin/python scripts/train_landmarks.py --config configs/demo_landmarks.yaml
PYTHONPATH=src .venv312/bin/python scripts/build_gallery.py --config configs/demo_gallery.yaml
PYTHONPATH=src .venv312/bin/python scripts/predict_image.py --config configs/demo_api.yaml --image data/demo/query/query_tabby.png
```

## 说明

- demo 配置启用了 `allow_full_image_fallback`
- 这是为了在合成图上跳过 Haar 检测失败的影响
- 默认正式配置仍然不会启用这个回退
