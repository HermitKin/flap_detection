import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam3_level_annotator.geometry import load_yolo_masks, mask_iou, masks_to_yolo_lines


class GeometryTest(unittest.TestCase):
    def test_yolo_round_trip(self) -> None:
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:81, 50:151] = True
        lines = masks_to_yolo_lines([mask], min_area=1, epsilon_ratio=0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].split()[0], "0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text(lines[0], encoding="utf-8")
            loaded = load_yolo_masks(path, mask.shape)
        self.assertEqual(len(loaded), 1)
        self.assertGreater(mask_iou(mask, loaded[0]), 0.97)

    def test_small_speck_is_removed(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        cv2.rectangle(mask.view(np.uint8), (5, 5), (6, 6), 1, -1)
        self.assertEqual(masks_to_yolo_lines([mask], min_area=40), [])

    def test_disconnected_islands_stay_one_instance(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:70, 10:50] = True
        mask[80:95, 80:95] = True
        self.assertEqual(len(masks_to_yolo_lines([mask], min_area=1)), 1)


if __name__ == "__main__":
    unittest.main()
