from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_compare_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_image_pixels.py"
    spec = importlib.util.spec_from_file_location("compare_image_pixels", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise unittest.SkipTest(f"optional image comparison dependencies unavailable: {exc}") from exc
    return module


class CompareImagePixelsTests(unittest.TestCase):
    def test_compares_pixel_data(self):
        module = load_compare_module()

        np = module.np
        reference = Path("reference.png")
        same = Path("same.png")
        different = Path("different.png")
        images = {
            reference: np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
            same: np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8),
            different: np.full((2, 2, 3), [10, 20, 31], dtype=np.uint8),
        }
        module.image_array = lambda path: images[path]
        report = module.compare_to_reference(reference, [same, different])

        self.assertTrue(report["comparisons"][0]["matches_reference"])
        self.assertEqual(report["comparisons"][0]["changed_pixels"], 0)
        self.assertFalse(report["comparisons"][1]["matches_reference"])
        self.assertEqual(report["comparisons"][1]["changed_pixels"], 4)


if __name__ == "__main__":
    unittest.main()
