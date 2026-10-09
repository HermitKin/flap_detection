# SAM2 液位计半自动分割标注工具（CPU）

这里提供两种本地标注界面：

`interactive_level_gauge_annotator.py` 采用与挂钩标注器一致的工作流：人工给每个液位计画矩形，SAM2 Tiny 生成精细轮廓，每次操作即时保存并支持断点恢复。

## 1. 环境安装

建议使用独立环境，按需要安装 CPU 或 GPU 版 PyTorch，然后安装标注工具依赖：

```powershell
python -m pip install -r sam3_level_annotator/requirements.txt
```

请自行准备兼容的 `sam2_t.pt` 权重，例如保存在：

```text
models/sam2_t.pt
```

## 2. 启动

### 推荐：矩形提示 + 即时保存

```powershell
cd flap_detection
python -m sam3_level_annotator.interactive_level_gauge_annotator `
  --images "D:\data\level_gauge\images" `
  --labels "D:\data\level_gauge\labels" `
  --model ".\models\sam2_t.pt" `
  --device cpu `
  --imgsz 640
```

图片目录下的 `train/val/test` 等子目录会自动映射到标签目录。每次新增实例、撤销或清空都会原子写入标签；切图和退出会即时写入 `level_gauge_annotator_progress.json`。重新运行会回到上次图片。仅仅浏览一张未标图不会错误生成空标签，确认是负样本时请按 `C`。

操作键：

- 鼠标左键拖矩形：分割并立即保存一个液位计实例。
- `N` / 右方向键：下一张；`P` / 左方向键：上一张。
- `Z`：删除最后一个实例并立即保存。
- `C`：清空本图并保存空标签（负样本）。
- `S`：手动保存；`Q` / `Esc`：保存后退出。

也可以使用启动脚本：

```powershell
.\sam3_level_annotator\run_interactive.ps1 `
  --images "D:\data\level_gauge\images" `
  --labels "D:\data\level_gauge\labels" `
  --model ".\models\sam2_t.pt"
```

CPU 速度不够时可将 `--imgsz 640` 改为 `--imgsz 512`，代价是轮廓精度略有下降。

## 4. 输出

每张图生成一个同名 `.txt`，每行格式为：

```text
0 x1 y1 x2 y2 ... xn yn
```

坐标已归一化到 `[0, 1]`，类别 `0` 为 `liquid_level_gauge`，可直接用于 Ultralytics YOLO 实例分割。空标注也会写空 `.txt`，表示该图是负样本。

标签目录同级会生成 `level_gauge_annotator_progress.json`，用于断点恢复；训练只使用 `.txt` 即可。
