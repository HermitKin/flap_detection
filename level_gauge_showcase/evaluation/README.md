# 识别与读数统计

运行：

```powershell
D:/Anaconda/python.exe X:/Work/project_flapsdetection/level_gauge_showcase/evaluation/evaluate_pipeline.py
```

脚本读取展示程序已经保存的三类 JSON：

- `outputs/object_detection/data`：目标检测；
- `outputs/keypoint_detection/keypoints`：关键点检测；
- `outputs/read_roi/data`：未透视和透视后的读数。

首次运行会生成 `ground_truth_readings.csv`。人工查看仪表后，在
`actual_reading` 列填写真实读数；不参与统计的行把 `include` 改为 `0`。
再次运行即可得到未透视和透视后两套 MAE、RMSE、满量程归一化误差、
±1%/±2%/±5% 满量程命中率，以及透视校正改善/恶化样本数。

输出位于 `evaluation/results/`：

- `report.md`：中文汇总报告；
- `summary.json`：机器可读汇总；
- `object_detection_details.csv`：逐图 TP/FP/FN/IoU；
- `keypoint_detection_details.csv`：逐目标置信度和透视可用状态；
- `reading_details.csv`：真实、未透视、透视后读数与差值。

注意：GUI 只保存成功完成的 ReadROI JSON，失败操作目前没有统一落盘，
因此仅靠已有文件不能得到严格的 ReadROI 成功率分母。正式测试应先固定
测试图片清单，并保证成功、拒检和异常均保存一条状态记录。
