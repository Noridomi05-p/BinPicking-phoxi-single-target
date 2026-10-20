import unittest

import numpy as np

from phoxi_yolo3d.duplicates import suppress_duplicate_masks
from phoxi_yolo3d.types import InstanceDetection, InstanceMeasurement, InstanceResult


def result(index, score, mask, class_id=4):
    return InstanceResult(
        InstanceDetection(index, class_id, str(class_id), score, (0, 0, 99, 99), mask),
        InstanceMeasurement(False, 'test', 0, int(mask.sum()), 0),
    )


class DuplicateTests(unittest.TestCase):
    def test_contained_duplicate_lower_confidence_is_removed(self):
        full = np.zeros((100, 100), dtype=bool)
        full[10:90, 10:90] = True
        part = np.zeros_like(full)
        part[10:40, 10:40] = True
        kept, rejected = suppress_duplicate_masks([result(1, .28, part), result(0, .63, full)], {})
        self.assertEqual([x.detection.instance_id for x in kept], [0])
        self.assertEqual(rejected[0].postprocess.reason, 'DUPLICATE_MASK')
        self.assertEqual(rejected[0].postprocess.duplicate_of_instance_id, 0)

    def test_two_touching_boxes_same_class_are_preserved(self):
        left = np.zeros((100, 100), dtype=bool)
        left[:, :50] = True
        right = ~left
        kept, rejected = suppress_duplicate_masks([result(0, .9, left), result(1, .8, right)], {})
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected, [])

    def test_different_classes_are_not_suppressed(self):
        mask = np.ones((100, 100), dtype=bool)
        kept, rejected = suppress_duplicate_masks([result(0, .9, mask, 2), result(1, .8, mask, 4)], {})
        self.assertEqual(len(kept), 2)
        self.assertEqual(rejected, [])


if __name__ == '__main__':
    unittest.main()
