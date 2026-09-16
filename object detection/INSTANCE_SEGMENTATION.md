# YOLO11 实例分割：航拍与自动驾驶数据集方案

本项目不再使用 VisDrone，也不再把检测框交给 SAM 生成伪标签。新的主线使用带有真实实例掩膜的数据集：

- 低空无人机车辆方向：EVD4UAV。
- 高空航拍/遥感备选：iSAID。
- 自动驾驶方向：BDD100K Instance Segmentation。
- 小规模自动驾驶备选：Cityscapes Fine Annotations。

航拍和车载视角的类别、尺度及成像分布差异很大，建议分别训练两个模型，不直接混合成一个数据集。

## 数据集选择

### EVD4UAV：低空无人机车辆实例分割首选

EVD4UAV 由真实 UAV 在 50m、70m、90m 高度采集，论文报告 6,284 张图像、90,886 个车辆实例，标注包含水平框、旋转框和实例掩膜。它与“无人机俯视车辆轮廓”任务最匹配。

- 论文：<https://arxiv.org/abs/2403.05422>
- DOI：<https://doi.org/10.1109/IV55156.2024.10588796>

目前检索到的可直接训练 COCO 整理版为第三方发布，包含 `instances_train.json`、`instances_val.json` 和对应图像：

- COCO 整理版：<https://huggingface.co/datasets/jeff-calderon/UAV_INST_DATA>

第三方整理版的类别、样本数量和原论文全集不完全一致，且数据卡没有给出完整许可文本。正式论文实验前应核对原始 EVD4UAV 许可、抽样检查掩膜，并同时引用原论文和整理版本。

如果下载后目录为：

```text
EVD4UAV_Final_COCO/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
└── images/
    ├── train/
    └── val/
```

转换和训练：

```powershell
python prepare_coco_seg_dataset.py `
  --train-images "raw/evd4uav/images/train" `
  --train-json "raw/evd4uav/annotations/instances_train.json" `
  --val-images "raw/evd4uav/images/val" `
  --val-json "raw/evd4uav/annotations/instances_val.json" `
  --output "datasets/evd4uav-yolo"

python check_seg_dataset.py --data datasets/evd4uav-yolo/data.yaml --split train --samples 40

python train_seg.py `
  --data datasets/evd4uav-yolo/data.yaml `
  --model yolo11m-seg.pt `
  --epochs 250 --imgsz 1024 --batch 2 `
  --project runs/segment --name evd4uav-yolo11m-seg
```

### iSAID：高空航拍/遥感实例分割备选

iSAID 包含 2,806 张高分辨率航拍图像、655,451 个实例和 15 类目标。类别包括飞机、船、储油罐、运动场、桥梁、大型车辆、小型车辆、直升机、港口等。它的图像主要来自 Google Earth 和卫星，不是典型低空 UAV 视频，因此适合做高空航拍小目标基线，不应把它描述成低空无人机数据。数据用于学术研究，禁止商业使用。

- 数据集主页：<https://captain-whu.github.io/iSAID/dataset.html>
- 官方预处理工具：<https://github.com/CAPTAIN-WHU/iSAID_Devkit>

官方开发工具会把大图切成 800×800 patch，并生成 COCO 格式的训练、验证 JSON。测试集真值不公开，因此本地实验使用 train/val。

预处理后应得到类似结构：

```text
iSAID_patches/
├── train/
│   ├── images/
│   └── instance_only_filtered_train.json
└── val/
    ├── images/
    └── instance_only_filtered_val.json
```

不同版本工具生成的 JSON 文件名可能略有差异，以实际文件为准。

转换为 YOLO 实例分割格式：

```powershell
python prepare_coco_seg_dataset.py `
  --train-images "raw/isaid/iSAID_patches/train/images" `
  --train-json "raw/isaid/iSAID_patches/train/instance_only_filtered_train.json" `
  --val-images "raw/isaid/iSAID_patches/val/images" `
  --val-json "raw/isaid/iSAID_patches/val/instance_only_filtered_val.json" `
  --output "datasets/isaid-yolo"
```

训练前检查：

```powershell
python check_seg_dataset.py --data datasets/isaid-yolo/data.yaml --split train --samples 40
python check_seg_dataset.py --data datasets/isaid-yolo/data.yaml --split val --samples 20
```

训练：

```powershell
python train_seg.py `
  --data datasets/isaid-yolo/data.yaml `
  --model yolo11m-seg.pt `
  --epochs 300 --imgsz 800 --batch 2 `
  --project runs/segment --name isaid-yolo11m-seg
```

## BDD100K：自动驾驶实例分割

BDD100K 包含多种天气、道路类型和昼夜条件，官方 10K 图像子集提供实例分割标注。官方下载页提供 `10K Images` 和 `Instance Segmentation` 两部分。

- 下载页：<https://bdd-data.berkeley.edu/download.html>
- 官方工具和格式：<https://github.com/bdd100k/bdd100k>

先使用 BDD100K 官方工具把 RLE 标注转换为 COCO：

```powershell
git clone https://github.com/bdd100k/bdd100k.git external/bdd100k
python -m pip install -e external/bdd100k

python -m bdd100k.label.to_coco -m ins_seg `
  -i raw/bdd100k/labels/ins_seg/rles/ins_seg_train.json `
  -o raw/bdd100k/annotations/ins_seg_train_coco.json

python -m bdd100k.label.to_coco -m ins_seg `
  -i raw/bdd100k/labels/ins_seg/rles/ins_seg_val.json `
  -o raw/bdd100k/annotations/ins_seg_val_coco.json
```

再转换为 YOLO 多边形标签。RLE 解码需要 `pycocotools>=2.0.7`：

```powershell
python -m pip install "pycocotools>=2.0.7"

python prepare_coco_seg_dataset.py `
  --train-images "raw/bdd100k/images/10k/train" `
  --train-json "raw/bdd100k/annotations/ins_seg_train_coco.json" `
  --val-images "raw/bdd100k/images/10k/val" `
  --val-json "raw/bdd100k/annotations/ins_seg_val_coco.json" `
  --output "datasets/bdd100k-yolo"
```

检查和训练：

```powershell
python check_seg_dataset.py --data datasets/bdd100k-yolo/data.yaml --split train --samples 40
python check_seg_dataset.py --data datasets/bdd100k-yolo/data.yaml --split val --samples 20

python train_seg.py `
  --data datasets/bdd100k-yolo/data.yaml `
  --model yolo11m-seg.pt `
  --epochs 200 --imgsz 960 --batch 2 `
  --project runs/segment --name bdd100k-yolo11m-seg
```

如果显存不足，先将模型换成 `yolo11s-seg.pt`，然后将 `imgsz` 降至 640；不要一开始删除小目标标签。

## Cityscapes 备选

Cityscapes 提供 5,000 张精细标注图像和 20,000 张粗标注图像，其中车辆与人员类别带实例级标注。它规模较小、标注质量高，适合先验证自动驾驶分割流程，但主要是欧洲城市白天和较好天气，场景多样性弱于 BDD100K。下载需要注册，许可仅允许非商业使用。

- 数据集主页：<https://www.cityscapes-dataset.com/>
- 数据说明：<https://www.cityscapes-dataset.com/dataset-overview/>

## 转换器行为

`prepare_coco_seg_dataset.py` 支持：

- COCO polygon：直接转换，适合 iSAID。
- COCO compressed/uncompressed RLE：解码后提取轮廓，适合 BDD100K。
- 非连续 COCO category ID 自动映射为连续 YOLO 类别 ID。
- `--classes` 按名称筛选类别，并重新编号。
- 默认用硬链接复用图像，跨磁盘时自动复制。
- 多个不连通轮廓按最短距离连接，以满足 YOLO 每个实例一条多边形标签的格式。
- 跳过 `iscrowd`、过小区域和无效掩膜，并输出统计信息。

不要在同一个输出目录重复运行。转换器会拒绝非空目录，避免标签重复或数据混杂。

## 独立验证

```powershell
python val_seg.py `
  --model runs/segment/isaid-yolo11m-seg/weights/best.pt `
  --data datasets/isaid-yolo/data.yaml `
  --imgsz 800

python val_seg.py `
  --model runs/segment/bdd100k-yolo11m-seg/weights/best.pt `
  --data datasets/bdd100k-yolo/data.yaml `
  --imgsz 960
```

最终比较时记录 Box mAP 与 Mask mAP，实例分割结论以 `Mask mAP50` 和 `Mask mAP50-95` 为主。
