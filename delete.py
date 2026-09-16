from pathlib import Path


# =========================
# 路径自己填写
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
TXT_FOLDER = PROJECT_ROOT / "object detection" / "datasets" / "flap_cleaned" / "labels"
IMAGE_FOLDER = PROJECT_ROOT / "object detection" / "datasets" / "flap_cleaned" / "images"


# =========================
# 支持的图片格式
# =========================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def clean_orphan_txt():
    txt_dir = Path(TXT_FOLDER)
    image_dir = Path(IMAGE_FOLDER)

    if not txt_dir.exists():
        print(f"TXT文件夹不存在：{txt_dir}")
        return

    if not image_dir.exists():
        print(f"图片文件夹不存在：{image_dir}")
        return

    # 获取图片文件名，不包含后缀
    image_names = {
        p.stem
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    deleted_files = []
    kept_files = []

    # 遍历所有txt文件
    for txt_path in txt_dir.glob("*.txt"):

        txt_name = txt_path.stem

        # 如果没有同名图片
        if txt_name not in image_names:
            txt_path.unlink()
            deleted_files.append(txt_path.name)
        else:
            kept_files.append(txt_path.name)

    # =========================
    # 输出结果
    # =========================

    print("\n========== 清理结果 ==========")

    if deleted_files:
        print("\n已删除以下TXT文件：")
        for filename in deleted_files:
            print(f"删除：{filename}")
    else:
        print("\n没有需要删除的TXT文件。")

    print("\n------------------------------")
    print(f"保留TXT数量：{len(kept_files)}")
    print(f"删除TXT数量：{len(deleted_files)}")
    print("==============================")


if __name__ == "__main__":
    clean_orphan_txt()
