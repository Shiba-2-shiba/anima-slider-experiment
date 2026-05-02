from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def pixel_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_to_reference(reference: Path, candidates: list[Path]) -> dict:
    ref = image_array(reference)
    rows = []
    ref_hash = pixel_hash(ref)
    for candidate in candidates:
        arr = image_array(candidate)
        if arr.shape != ref.shape:
            rows.append(
                {
                    "path": str(candidate),
                    "pixel_hash": pixel_hash(arr),
                    "shape": list(arr.shape),
                    "matches_reference": False,
                    "shape_mismatch": True,
                }
            )
            continue
        diff = np.abs(arr.astype(np.int16) - ref.astype(np.int16))
        rows.append(
            {
                "path": str(candidate),
                "pixel_hash": pixel_hash(arr),
                "shape": list(arr.shape),
                "matches_reference": bool(np.array_equal(arr, ref)),
                "shape_mismatch": False,
                "max_abs_diff": int(diff.max()),
                "sum_abs_diff": int(diff.sum()),
                "changed_pixels": int(np.any(diff, axis=2).sum()),
            }
        )
    return {
        "reference": {
            "path": str(reference),
            "pixel_hash": ref_hash,
            "shape": list(ref.shape),
        },
        "comparisons": rows,
    }


def main(args):
    reference = Path(args.reference)
    candidates = [Path(path) for path in args.images]
    report = compare_to_reference(reference, candidates)
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.fail_on_difference and any(not row["matches_reference"] for row in report["comparisons"]):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare image pixel data while ignoring PNG metadata.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_difference", action="store_true")
    main(parser.parse_args())
