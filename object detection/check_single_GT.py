import cv2
import matplotlib.pyplot as plt

# 1. 图像路径 (请确保图片和代码在同一目录下，或者修改为绝对路径)
img_path = '3000242_jpg.rf.8c45bdd5617239b152d512933a2b08e6.jpg'

# 2. 你的标签数据 (YOLO格式: class x_center y_center width height)
label_data = """
0 0.5328125 0.4703125 0.11875 0.103125
0 0.63671875 0.7203125 0.1328125 0.0796875
"""


def visualize_gt(img_path, label_data):
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法找到图片 {img_path}")
        return

    # 获取图片尺寸
    h, w, _ = img.shape
    print(f"图片尺寸: 宽={w}, 高={h}")

    # 解析标签并画框
    lines = label_data.strip().split('\n')
    for line in lines:
        parts = list(map(float, line.split()))
        class_id = int(parts[0])
        x_center, y_center, width, height = parts[1:]

        # --- 核心逻辑：YOLO归一化坐标 转 像素坐标 ---
        # 1. 反归一化
        x_c_pixel = x_center * w
        y_c_pixel = y_center * h
        box_w_pixel = width * w
        box_h_pixel = height * h

        # 2. 计算左上角 (x1, y1) 和 右下角 (x2, y2)
        x1 = int(x_c_pixel - box_w_pixel / 2)
        y1 = int(y_c_pixel - box_h_pixel / 2)
        x2 = int(x_c_pixel + box_w_pixel / 2)
        y2 = int(y_c_pixel + box_h_pixel / 2)

        print(f"类别: {class_id}, 坐标: ({x1}, {y1}), ({x2}, {y2})")

        # 3. 画矩形框 (绿色, 线宽2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 4. 添加标签文字
        text = f"ID: {class_id}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # 显示结果 (Matplotlib 显示需要转 RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')  # 不显示坐标轴
    plt.title("Ground Truth Visualization")
    plt.show()


# 运行函数
if __name__ == "__main__":
    visualize_gt(img_path, label_data)