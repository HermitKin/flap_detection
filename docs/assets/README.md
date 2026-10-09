# 展示素材来源

素材用于公开项目说明，不包含业务数据集或内部现场照片。

| 文件 | 来源 |
| --- | --- |
| `ui-detection.png` | `app.py --demo` 实际目标页窗口截图 |
| `ui-keypoints.png` | 同一应用关键点页截图 |
| `ui-reading.png` | 同一应用实际算法读数页截图 |
| `ui-hsv.png` | HSV 中间结果窗口截图 |
| `pipeline.png` / `pipeline.gif` | 合成几何 + 实际校正和读数的阶段展示 |
| `level-sweep.gif` | 每帧重算的合成液位序列，非实时模型推理 |
| `architecture.png` | 系统结构图 |
| `algorithm.png` | 当前算法计算的状态曲线与公式 |
| `synthetic-results.json` | 六组合成案例的计算值，不含绝对路径 |

## 复现

从仓库根目录运行：

```bash
python -m pip install -r requirements-demo.txt
python tools/build_showcase_assets.py
python level_gauge_showcase/app.py --demo
```

生成脚本只读取自己构造的图像，在临时目录中计算，不读取数据集、权重或私有运行产物。图形使用 Pillow 绘制，GIF 使用共享调色板循环播放。没有通过图像生成模型编造界面。

截图来自实际桌面窗口。发布前检查窗口没有其他应用遮挡、个人路径或文件对话框。合成案例的框与关键点不代表模型性能。
