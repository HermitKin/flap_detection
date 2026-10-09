"""Tests exercise production reading on known synthetic geometry, without YOLO."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "level_gauge_showcase"))
from direct_hsv_reader import analyze_direct_flap_strip
from synthetic_demo import make_gauge, run_example
from services import ObjectDetectionService, ReadingService


class PublicDemoTests(unittest.TestCase):
    def test_known_levels_and_two_colours(self):
        with tempfile.TemporaryDirectory() as directory:
            for colour in ("red", "blue"):
                for level in (0.35, 0.62, 0.78):
                    with self.subTest(colour=colour, level=level):
                        example = run_example(Path(directory), level, colour)
                        report = example["record"].reports[0]
                        self.assertTrue(report["perspective_applied"])
                        self.assertLess(abs(report["reading"] - level * 100), 1.5)

    def test_nonzero_scale_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            example = run_example(Path(directory), 0.62)
            record = ReadingService(ObjectDetectionService()).read(
                example["source"], 20, 220, False, existing_quads=[example["quad"]])
            self.assertLess(abs(record.reports[0]["reading"] - 144), 3)

    def test_uniform_region_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_direct_flap_strip(np.full((560, 160, 3), 220, np.uint8))

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            ReadingService(ObjectDetectionService()).read(Path("unused.png"), 100, 0, False)

    def test_synthetic_domain(self):
        with self.assertRaises(ValueError):
            make_gauge(0.99)


if __name__ == "__main__":
    unittest.main()
