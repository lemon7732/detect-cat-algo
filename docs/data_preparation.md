# 数据准备说明

## 二分类数据

### 方案 A：本地目录

将公开数据集整理为：

```text
data/binary/raw/cat/
data/binary/raw/not_cat/
```

建议：
- `cat/` 使用猫正样本图像
- `not_cat/` 组合狗、人、鸟等非猫图像
- 文件名无需额外携带标签

训练时会自动：
- 按最大边 `300` 像素等比缩放
- 输出到 `data/binary/processed`
- 再按配置切分训练集和验证集

### 方案 B：`TensorFlow Datasets`

仓库已支持两组公开数据集直接作为正式训练源：

- `cats_vs_dogs`
- `oxford_iiit_pet`
- `celeb_a`
- `caltech_birds2011`

对应配置：

- [configs/binary_cats_vs_dogs.yaml](/home/lemon/self/code/algo/configs/binary_cats_vs_dogs.yaml)
- [configs/binary_oxford_iiit_pet.yaml](/home/lemon/self/code/algo/configs/binary_oxford_iiit_pet.yaml)
- [configs/binary_thesis_public.yaml](/home/lemon/self/code/algo/configs/binary_thesis_public.yaml)

可以直接下载：

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets cats_vs_dogs oxford_iiit_pet celeb_a caltech_birds2011
```

下载状态检查：

```bash
PYTHONPATH=src python scripts/check_downloads.py
```

第一次训练时会自动下载到 `data.tfds_data_dir`，然后导出到 `data.processed_dir` 继续走现有训练流程。

建议：
- `cats_vs_dogs` 作为主二分类训练集
- `oxford_iiit_pet` 适合作为补充，尤其是开启 `use_head_bbox: true` 时更接近猫脸任务
- 若想更贴近论文的 `cat vs dog/bird/human` 场景，直接使用 [binary_thesis_public.yaml](/home/lemon/self/code/algo/configs/binary_thesis_public.yaml)

## 关键点数据

### 方案 A：`CAT Dataset` 原生 9 点标注

这是当前最贴近论文九点关键点训练描述的公开数据源。推荐使用 Kaggle 上的 `crawford/cat-dataset`，其原始标注通常与图片同名，扩展名为 `.cat`。

推荐目录：

```text
data/cat_dataset/
  CAT_00/
    00000001_000.jpg
    00000001_000.jpg.cat
```

对应配置：

- [configs/landmarks_cat_dataset.yaml](/home/lemon/self/code/algo/configs/landmarks_cat_dataset.yaml)

下载：

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets cat_dataset
```

训练前会自动：
- 递归扫描图片与 `.cat` 标注
- 解析原生 9 点坐标
- 用关键点自动生成脸框并裁剪
- 将关键点按裁剪框归一化

注意：
- `.cat` 标注首个数字通常是关键点数量，后续是 9 点的 `x y` 坐标
- 工程默认按论文使用的 9 点顺序读取
- 该方案比 `CatFLW 48->9` 更适合作为论文主训练源

### 方案 B：本地 `CSV + images`

将猫脸关键点数据放到：

```text
data/landmarks/images/
data/landmarks/train.csv
```

`train.csv` 推荐字段：

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

训练前会自动：
- 尝试用 Haar 检测猫脸并裁剪
- 将关键点按脸框归一化
- 输出裁剪图到 `data/landmarks/processed`

### 方案 C：`CatFLW`

`CatFLW` 是公开猫脸 landmark 数据集，仓库已经支持直接解析本地解压后的标注文件。

推荐目录：

```text
data/catflw/
  images/
  labels/
```

对应配置：

- [configs/landmarks_catflw.yaml](/home/lemon/self/code/algo/configs/landmarks_catflw.yaml)

`CatFLW` 原始标注通常多于论文里的 9 点，所以工程里把“48 点到 9 点”的映射做成了 `data.point_groups` 配置。每一组表示一个论文关键点由哪些源点平均得到。

注意：
- 你需要先确认自己下载版本的 landmark 顺序，再调整 `point_groups`
- 如果标注是单个 `annotation.json`，可改用 `data.annotation_json`
- 如果标注里已经有 face bbox，预处理会优先使用标注框；否则回退到 Haar 检测或整图
- 当前更推荐把 `CatFLW` 作为补充数据，而不是论文主数据源

## 个体图库 / 底库

如果公开个体数据或校园猫照片已经按“每只猫一个文件夹”整理，可以直接生成 `metadata.csv`：

```bash
PYTHONPATH=src python scripts/generate_gallery_metadata.py --gallery-root data/gallery --output data/gallery/metadata.csv
```

这适合：
- 自己采集的校园猫图片
- `CatIndividualImages` 这类按个体组织的公开数据

如果已经安装 `wildlife-datasets`，也可以尝试直接下载：

```bash
PYTHONPATH=src python scripts/download_public_datasets.py --datasets cat_individual_images
```

同样可以用下面的命令统一检查 `TFDS`、`CatFLW` 和 `CatIndividualImages` 是否已经完成：

```bash
PYTHONPATH=src python scripts/check_downloads.py
```

如果你还需要检查 `CAT Dataset`，该命令现在也会一起返回 `cat_dataset` 的状态。

## 校园猫底库

准备 `data/gallery/metadata.csv`，最小字段：

- `cat_id`
- `name`
- `sex`
- `age`
- `description`
- `image_path`

其中 `image_path` 可以写相对路径或绝对路径。每只猫建议至少 5-10 张清晰正脸图。
