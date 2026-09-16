import os
import cv2
from pathlib import Path


# =========================
# 路径自己填写
# =========================

IMAGE_DIR = r""   # 原始图片目录
LABEL_DIR = r""   # YOLO检测标签目录
SAVE_DIR = r""    # 裁剪后图片保存目录


# =========================
# 参数
# =========================

# 只裁指定类别
# None = 所有类别
# 例如只裁 class 0：TARGET_CLASS = 0
TARGET_CLASS = None

# 裁剪框额外扩展比例
# 0.00 = 完全按照GT框
# 0.05 = 上下左右各扩展5%
PADDING = 0.00

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}


def crop_detection_gt():

    image_dir = Path(IMAGE_DIR)
    label_dir = Path(LABEL_DIR)
    save_dir = Path(SAVE_DIR)

    save_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_crops = 0

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        total_images += 1

        # 对应YOLO标签
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[跳过] 无标签：{image_path.name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[错误] 无法读取图片：{image_path}")
            continue

        img_h, img_w = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]

        for obj_id, line in enumerate(lines):

            data = line.split()

            if len(data) < 5:
                print(f"[标签格式错误] {label_path}: {line}")
                continue

            class_id = int(float(data[0]))

            # 类别过滤
            if TARGET_CLASS is not None and class_id != TARGET_CLASS:
                continue

            # YOLO归一化坐标
            xc = float(data[1])
            yc = float(data[2])
            bw = float(data[3])
            bh = float(data[4])

            # 转换成像素坐标
            xc *= img_w
            yc *= img_h
            bw *= img_w
            bh *= img_h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            # 扩展边缘
            pad_x = bw * PADDING
            pad_y = bh * PADDING

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # 防止越界
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(img_w, int(round(x2)))
            y2 = min(img_h, int(round(y2)))

            if x2 <= x1 or y2 <= y1:
                print(f"[无效框] {image_path.name} object={obj_id}")
                continue

            # 裁剪
            crop = image[y1:y2, x1:x2]

            # 保存名称
            save_name = (
                f"{image_path.stem}"
                f"_cls{class_id}"
                f"_{obj_id}"
                f"{image_path.suffix}"
            )

            save_path = save_dir / save_name

            cv2.imwrite(str(save_path), crop)

            total_crops += 1

    print("\n==============================")
    print(f"处理原图：{total_images}")
    print(f"裁剪目标：{total_crops}")
    print(f"保存位置：{SAVE_DIR}")
    print("==============================")


if __name__ == "__main__":
    crop_detection_gt()