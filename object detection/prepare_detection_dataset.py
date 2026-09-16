from pathlib import Path
import random


ROOT = Path("/workspace/datasets/flap_data")
SOURCE_IMAGES = ROOT / "all_images"
SOURCE_LABELS = ROOT / "all_labels"
SEED = 42
TRAIN_RATIO = 0.8


def main() -> None:
    images = sorted(
        path for path in SOURCE_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        raise RuntimeError(f"No images found in {SOURCE_IMAGES}")

    missing = [path.name for path in images if not (SOURCE_LABELS / f"{path.stem}.txt").is_file()]
    if missing:
        raise RuntimeError(f"Missing labels for {len(missing)} images; first: {missing[:5]}")

    shuffled = images.copy()
    random.Random(SEED).shuffle(shuffled)
    split_at = int(len(shuffled) * TRAIN_RATIO)
    splits = {"train": shuffled[:split_at], "val": shuffled[split_at:]}

    for split, split_images in splits.items():
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            image_link = image_dir / image.name
            label = SOURCE_LABELS / f"{image.stem}.txt"
            label_link = label_dir / label.name
            if image_link.is_symlink() or image_link.exists():
                image_link.unlink()
            if label_link.is_symlink() or label_link.exists():
                label_link.unlink()
            image_link.symlink_to(image)
            label_link.symlink_to(label)

        print(f"{split}: {len(split_images)} images")


if __name__ == "__main__":
    main()
