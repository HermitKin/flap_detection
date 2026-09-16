import cv2
import os
import random
import glob

# ================= 配置区域 (请修改这里) =================
# 1. 图片文件夹路径
IMG_DIR = r'dataset/dataset_solarpanel/val/images'
# 2. 标注文件夹路径 (txt文件所在位置)
LABEL_DIR = r'dataset/dataset_solarpanel/val/labels'
# 3. 结果保存的新文件夹路径 (会自动创建)
OUTPUT_DIR = r'solarpanel_results_batch'

# VisDrone 类别名称
CLASS_NAMES = {
    0: 'solar-panel'
}
# ===========================================
# 高亮霓虹色卡 (BGR格式)
BRIGHT_COLORS = [
    (0, 255, 0),      # 荧光绿

]
# BRIGHT_COLORS = [
#     (255, 0, 255),    # 洋红
#     (255, 255, 0),    # 青色
#     (0, 255, 255),    # 亮黄
#     (0, 255, 0),      # 荧光绿
#     (0, 165, 255),    # 橙色
#     (255, 191, 0),    # 深天蓝
#     (147, 20, 255),   # 深粉
#     (255, 50, 50),    # 亮蓝
#     (50, 205, 50),    # 酸橙绿
#     (180, 105, 255),  # 热粉
#     (211, 0, 148),    # 紫罗兰
#     (200, 200, 200)   # 浅灰
# ]
def get_bright_color(cls_id):
    return BRIGHT_COLORS[cls_id % len(BRIGHT_COLORS)]

def draw_clean_box(img, line):
    h_img, w_img, _ = img.shape
    parts = line.strip().split()
    if len(parts) < 5: return

    try:
        cls_id = int(parts[0])
        x_c = float(parts[1])
        y_c = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return

    # 坐标转换 (YOLO归一化 -> 像素坐标)
    x_min = int((x_c - 0.5 * w) * w_img)
    y_min = int((y_c - 0.5 * h) * h_img)
    x_max = int((x_c + 0.5 * w) * w_img)
    y_max = int((y_c + 0.5 * h) * h_img)

    color = get_bright_color(cls_id)

    # === 修改点 ===
    # 1. thickness=1 (最细线条)
    # 2. 删除了 cv2.putText 和 label 背景绘制代码
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)

def batch_process_clean():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取 jpg 和 png
    img_files = glob.glob(os.path.join(IMG_DIR, '*.jpg')) + \
                glob.glob(os.path.join(IMG_DIR, '*.png'))
    img_files.sort()
    
    print(f"开始处理 {len(img_files)} 张图片 (模式: 仅边框，细线条)...")

    for i, img_path in enumerate(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(LABEL_DIR, basename + '.txt')

        img = cv2.imread(img_path)
        if img is None: continue

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    draw_clean_box(img, line)

        save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, img)

        if (i + 1) % 50 == 0:
            print(f"进度: {i + 1}/{len(img_files)}")

    print(f"完成! 结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    batch_process_clean()