# 服务器批量关键点检测与读数

服务器目录示例：`/host/project/level_gauge_batch`

登录后手动进入正在运行的 YOLO 容器：

```bash
cd /host/project/level_gauge_batch
docker exec -it yolo_container bash
```

进入容器后，手动启动批量 Python：

```bash
cd /workspace/level_gauge_batch
python remote_batch_reading.py
```

默认配置：

- 数据集：`/workspace/datasets/flap_keypoint_full/images/test`
- 关键点模型：`/workspace/runs/pose/flap_pose4_with_yolo11s-fullset/weights/best.pt`
- 输入尺寸：960
- 置信度：0.20
- GPU：0
- 输出：`/workspace/level_gauge_batch/batch_results`
- 读数算法：与部署时 `services.py` 的 App 入口一致

每张图会询问量程。输入 `q` 会保存进度并退出，输入 `s` 会跳过当前图。
再次运行默认跳过已有 `result.json` 的图片，所以可以分多次完成。

输出结构：

```text
batch_results/
├─ reading_statistics.csv
├─ batch_summary.json
└─ 图片名称/
   ├─ pose_detected.png
   ├─ pose_keypoints.json
   ├─ result.json
   └─ gauge_01/
      ├─ 01_roi.png
      ├─ 02_unrectified_reading.png
      ├─ 03_rectified.png
      ├─ 04_reading_result.png
      ├─ 05_colour_interval.png
      └─ 06_hsv.png
```

`reading_statistics.csv` 包含图片名、空白人工真实读数、量程、Pose 置信度、
未透视读数、透视后读数、两者差值、失败原因和对应 JSON 路径。人工填写的
`actual_reading` 在续跑和重新汇总时会保留。

可切换关键点模型：

```bash
python remote_batch_reading.py --pose-model /workspace/runs/pose/其他模型/weights/best.pt
```

可显式选择读数算法：

```bash
python remote_batch_reading.py --reader strict
python remote_batch_reading.py --reader legacy
```

从指定文件开始：

```bash
python remote_batch_reading.py --start flap_20260817_0500__i00.jpg
```

若已准备批量量程表，可使用列名 `image_name,scale_min,scale_max`：

```bash
python remote_batch_reading.py --range-csv /workspace/level_gauge_batch/ranges.csv
```

## 单张图片

进入容器后直接运行：

```bash
python single_image_reading.py
```

程序会依次询问图片路径、量程下限和量程上限。也可以一次传入：

```bash
python single_image_reading.py \
  /workspace/datasets/flap_keypoint_full/images/test/flap_20260817_0183__i00.jpg \
  --scale-min 0 \
  --scale-max 100
```
