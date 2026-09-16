# Flap Detection

磁翻板液位计目标检测、关键点检测、透视校正与液位读数相关代码。

## 主要内容

- `object detection/`：基于 Ultralytics YOLO 的训练与推理代码。
- `level_gauge_showcase/`：液位计检测和读数演示应用。
- `sam3_level_annotator/`：交互式标注工具。
- 根目录 Python 脚本：数据准备、评估、校正、可视化与单图推理工具。

## 环境

可使用 `environment.yml` 创建 Conda 环境；演示应用和标注工具也分别提供了 `requirements.txt`。

## 本地资源

仓库不包含数据集、标注结果、训练输出、模型权重或本地运行产物。运行训练或推理前，请按脚本参数或配置文件准备对应数据和模型文件。

## 许可

`object detection/` 中的 Ultralytics 衍生代码沿用其 AGPL-3.0 许可，详见该目录内的 `LICENSE`。

