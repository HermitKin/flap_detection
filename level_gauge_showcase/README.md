# 磁翻板液位计完整流程展示

项目完整展示与安装路线见 [项目首页](../README.md)，按钮操作见 [使用指南](../docs/USAGE.md)，原理见 [算法解析](../docs/ALGORITHM.md)。

无需模型权重即可体验公开合成案例：`python level_gauge_showcase/app.py --demo`（从仓库根目录运行）。框和四角来自已知几何，透视校正与读数调用本项目实际代码。轻量依赖为根目录 `requirements-demo.txt`。

本目录是一份独立展示程序，包含目标检测、关键点检测和 ReadROI 读数三个页面。旧版核心代码保存在 `legacy/`，展示程序通过 `services.py` 复用关键点数据格式和液位计算逻辑。目标检测和关键点检测在隔离的 Python 子进程中运行，主界面进程只负责显示、OpenCV 图像处理和读数，从而避免 Windows 下 PyTorch 与 OpenCV 的 OpenMP 运行库冲突。

## 启动

在 PowerShell 或 CMD 中运行：

```powershell
python level_gauge_showcase/app.py
```

依赖：`ultralytics`、`opencv-python`、`numpy`、`Pillow`。模型首次调用时会有短暂加载时间。

如当前 Python 环境尚未安装依赖，可执行：

```powershell
python -m pip install -r level_gauge_showcase/requirements.txt
```

## 三个页面

1. **液柱计检测**：选择单图或多图，后台逐张调用目标检测模型；每张图检测后会自动保存检测图、JSON，并使用原始像素按目标框自动裁剪和保存 ROI。“查看检测结果”会弹出结果图、检测框位置、类别和置信度。
2. **关键点检测**：选择和查看方式相同，也可通过“读取第1页 ROI 结果”直接读取当前会话中第一页的检测记录和全部已裁剪 ROI，不会弹出文件选择框；每张图片检测完成后会立即自动保存标注图、关键点 JSON 和目标 ROI。“查看检测结果”会弹出结果图、检测框和每个关键点的坐标及置信度，四点顺序为 `TL, TR, BR, BL`。
3. **ReadROI 读数**：第二页只展示带框线和关键点的标注图；进入第三页时直接读取第二页按关键点检测框保存的无标注 ROI，并将四点转换为裁剪图局部坐标。透视校正后，程序在标准矩形完整宽度上逐列从底部追踪循环 Hue、S、V 状态，组合分界高度相近且横向连续的列，从而自适应得到翻片颜色区域，不再使用固定中心或固定宽度。短小间隙会被闭运算连接，状态断开处作为液面分界。标准矩形总高度只作为手动量程的换算标尺；检测框、颜色裁剪图和 HSV 图只输出从分界位置到底部的连续有色示数段。第三页同时显示未透视对比、透视校正和正式读数结果。

第一页和第二页均可点击“选择模型”临时切换推理权重，也可点击“恢复默认模型”。切换只影响之后启动的检测，结果 JSON 会记录实际使用的模型路径。第三页选择“重新调用关键点模型并透视校正”时，会使用第二页当前选中的关键点模型。

界面显示只生成缩略预览，不会修改输入图片尺寸；保存的检测结果保持原始分辨率。

## 输出目录

```text
outputs/
├─ object_detection/
│  ├─ detected_images/
│  ├─ data/
│  └─ roi_crops/
├─ keypoint_detection/
│  ├─ detected_images/
│  ├─ keypoints/
│  └─ roi_crops/
└─ read_roi/
   ├─ input_roi/
   ├─ detection_results/
   ├─ images/
   ├─ unrectified_comparison/
   ├─ rectified_images/
   ├─ flap_color_strips/
   ├─ hsv_results/
   └─ data/
```

## 代码组成

- `app.py`：Tkinter 展示界面与交互。
- `services.py`：模型调用、数据结构、裁剪、读数及保存。
- `direct_hsv_reader.py`：优先使用的全宽逐列、底部连续 HSV 方法。
- `legacy_candidate_reader.py`：未接入 GUI 的旧版76候选条带变化峰算法，可单独运行或手动替换当前算法。
- `config.py`：模型路径、输出路径和默认参数。
- `models/`：展示所需的目标检测与关键点权重副本。
- `legacy/`：本项目原有脚本的副本，用于保留和对照已有实现。

## 独立运行旧版 76 候选算法

该文件是“加入从底部开始、颜色连续、断开/重现检查及横向稳定宽度约束之前”的版本。它不会被当前 GUI 自动调用，也不会在当前算法失败时自动回退。

不带参数运行时会弹出图片选择框，并在命令行询问量程：

```powershell
python level_gauge_showcase/legacy_candidate_reader.py
```

也可以全部通过命令行传入：

```powershell
python level_gauge_showcase/legacy_candidate_reader.py "ROI图片路径" --scale-min 0 --scale-max 100 --output-dir "输出目录"
```

输入应为第一页/第二页得到的液位计 ROI，或已经透视校正的 ROI，而不是包含大范围背景的场景原图。脚本保存 `detection_result.png`、仅含有色示数段的 `colour_interval.png` 和 `reading.json`。

如需临时让 GUI 使用此旧版，只修改 `services.py` 的导入：

```python
# 当前严格版
from direct_hsv_reader import analyze_direct_flap_strip

# 旧版对照（将上一行替换为这一行）
from legacy_candidate_reader import analyze_direct_flap_strip
```
