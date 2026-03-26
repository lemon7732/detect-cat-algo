# API 说明

## 启动服务

```bash
PYTHONPATH=src python scripts/serve_api.py --config configs/api.yaml
```

说明：
- 若只需要接口结构和 OpenCV 检测，安装 `requirements.txt` 即可
- 若需要真实二分类/关键点模型推理，仍需安装 `requirements-ml.txt` 并使用 TensorFlow 兼容解释器

## 接口

### `GET /health`

健康检查。

### `POST /predict/species`

上传图片后返回：
- `is_cat`
- `cat_probability`

### `POST /detect/cat-face`

上传图片后返回：
- `faces`
- `primary_bbox`

### `POST /predict/landmarks`

上传图片后返回：
- `bbox`
- `landmarks`
- `normalized_landmarks`

### `POST /identify/cat`

上传图片后返回：
- `matched_cat_id`
- `matched_name`
- `cosine_score`
- `euclidean_distance`
- `top_k`
- `is_unknown`

### `POST /gallery/rebuild`

重建特征底库并刷新服务内存缓存。
