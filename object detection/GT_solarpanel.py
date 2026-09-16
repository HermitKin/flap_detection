import cv2
import os
import glob
import numpy as np

# ================= 配置区域 =================
# 1. 图片文件夹路径
IMG_DIR = r'dataset/dataset_solarpanel/test/images'
# 2. 标注文件夹路径
LABEL_DIR = r'dataset/dataset_solarpanel/test/labels'
# 3. 结果保存路径
OUTPUT_DIR = r'vis_results_solar'

# ===========================================

# 高亮霓虹色 (BGR格式) - 只有一类 Solar Panel，用最显眼的颜色
# 比如：亮黄 或 荧光绿
BOX_COLOR = (0, 255, 0)  # 亮lv色


def draw_bounding_box_from_polygon(img, line):
    h_img, w_img, _ = img.shape

    parts = line.strip().split()
    # 至少要有 class_id 和 x1, y1
    if len(parts) < 3: return

    try:
        # parts[0] 是 class_id，后面全是坐标
        # 提取所有坐标转化为 float
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return

    # 分离 X 和 Y 坐标
    # 格式是 x1, y1, x2, y2, ...
    x_coords = coords[0::2]  # 取偶数索引: 0, 2, 4...
    y_coords = coords[1::2]  # 取奇数索引: 1, 3, 5...

    if not x_coords or not y_coords:
        return

    # === 核心逻辑：计算外接矩形 ===
    # 找到多边形所有点中，最小的 x/y 和最大的 x/y
    x_min_norm = min(x_coords)
    x_max_norm = max(x_coords)
    y_min_norm = min(y_coords)
    y_max_norm = max(y_coords)

    # 反归一化：转回像素坐标
    x1 = int(x_min_norm * w_img)
    y1 = int(y_min_norm * h_img)
    x2 = int(x_max_norm * w_img)
    y2 = int(y_max_norm * h_img)

    # 绘制矩形框
    # thickness=1 (细线条)
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 1)


def batch_process_box():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 支持多种图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(IMG_DIR, ext)))

    img_files.sort()

    print(f"开始处理 {len(img_files)} 张图片 (将多边形转换为矩形框)...")

    for i, img_path in enumerate(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(LABEL_DIR, basename + '.txt')

        img = cv2.imread(img_path)
        if img is None: continue

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    draw_bounding_box_from_polygon(img, line)

        save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, img)

        if (i + 1) % 50 == 0:
            print(f"进度: {i + 1}/{len(img_files)}")

    print(f"处理完成! 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    batch_process_box()