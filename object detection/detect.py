import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/yolo11-HFA+DEConv-tbar/weights/best.pt') # select your model.pt path
    model.predict(source='features_dir/tbar',
                  imgsz=640,
                  project='runs/detect/',
                  name='yolo11-HFA+DEConv_FM-tbar',
                  line_width=1,
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes默认是2
                  # show_conf=False, # do not show prediction confidence 是否显示置信度
                  # show_labels=False, # do not show prediction labels  是否显示标签
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )