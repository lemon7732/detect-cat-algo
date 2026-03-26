# 推理与建库说明

## 建库

```bash
PYTHONPATH=src python scripts/build_gallery.py --config configs/gallery.yaml
```

输出：
- `artifacts/gallery/gallery_index.json`
- `artifacts/gallery/gallery_vectors.npz`
- `artifacts/gallery/gallery_failures.json`

如果图库目录已经是：

```text
data/gallery/
  cat_001/
  cat_002/
```

可以先自动生成元数据：

```bash
PYTHONPATH=src python scripts/generate_gallery_metadata.py --gallery-root data/gallery --output data/gallery/metadata.csv
```

建库流程：
1. 猫/非猫二分类
2. 猫脸检测
3. 九点关键点提取
4. 生成 18 维特征向量
5. 对同一只猫聚合均值原型向量

## 单图识别

```bash
PYTHONPATH=src python scripts/predict_image.py --config configs/api.yaml --image /path/to/image.jpg
```

## 批量识别

```bash
PYTHONPATH=src python scripts/predict_batch.py --config configs/api.yaml --input-dir /path/to/images
```
