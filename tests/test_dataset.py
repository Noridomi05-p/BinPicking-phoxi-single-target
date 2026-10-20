import tempfile
import unittest
from pathlib import Path

from phoxi_yolo3d.dataset import (
    parse_yolo_segmentation,
    prepare_dataset,
    validate_raw_labels,
    write_yolo_segmentation,
)


class DatasetTests(unittest.TestCase):
    def test_round_trip_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            polygon = [0.1, 0.1, 0.8, 0.1, 0.8, 0.8, 0.1, 0.8]
            for index in range(10):
                (images / f"image_{index:02d}.png").write_bytes(b"fake")
                write_yolo_segmentation(labels / f"image_{index:02d}.txt", [(index % 2, polygon)])
            parsed = parse_yolo_segmentation(labels / "image_00.txt")
            self.assertEqual(parsed[0][0], 0)
            report = validate_raw_labels(images, labels, ["a", "b"])
            self.assertFalse(report["errors"])
            prepared = prepare_dataset(
                images,
                labels,
                root / "dataset",
                ["a", "b"],
                ratios=(0.7, 0.2, 0.1),
                seed=7,
            )
            self.assertEqual(prepared["splits"]["train"]["images"], 7)
            self.assertEqual(prepared["splits"]["val"]["images"], 2)
            self.assertEqual(prepared["splits"]["test"]["images"], 1)
            self.assertTrue((root / "dataset" / "dataset.yaml").exists())

    def test_invalid_polygon_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "bad.txt"
            label.write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_yolo_segmentation(label)


if __name__ == "__main__":
    unittest.main()

