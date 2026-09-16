import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam3_level_annotator.interactive_level_gauge_annotator import (
    atomic_write_text,
    load_polygons,
    save_polygons,
)


class AnnotationIoTest(unittest.TestCase):
    def test_polygon_label_round_trip(self) -> None:
        polygon = np.asarray([[10, 20], [80, 20], [80, 90], [10, 90]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels" / "train" / "sample.txt"
            save_polygons(path, [polygon], (100, 100))
            loaded = load_polygons(path, (100, 100))
        self.assertEqual(len(loaded), 1)
        np.testing.assert_allclose(loaded[0], polygon, atol=1e-4)

    def test_empty_negative_label_and_atomic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "labels" / "negative.txt"
            save_polygons(label, [], (100, 100))
            self.assertTrue(label.exists())
            self.assertEqual(label.read_text(encoding="ascii"), "")
            progress = root / "progress.json"
            atomic_write_text(progress, '{"index": 7}\n')
            self.assertEqual(progress.read_text(encoding="utf-8"), '{"index": 7}\n')
            self.assertFalse(progress.with_name(progress.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
