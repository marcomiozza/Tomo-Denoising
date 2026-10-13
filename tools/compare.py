#!/usr/bin/env python3
"""
Pixel-by-pixel comparison of two DICOM folders with the same file names.

Checks that an output of the project matches a reference, for example a set
of results produced earlier.

Usage:
    python tools/compare.py <new_folder> <reference_folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pydicom


def main():
    ap = argparse.ArgumentParser(description="Pixel-by-pixel comparison of two DICOM folders.")
    ap.add_argument("new")
    ap.add_argument("reference")
    args = ap.parse_args()

    new_files = sorted(Path(args.new).glob("*.dcm"))
    if not new_files:
        raise SystemExit(f"No .dcm files in {args.new}")
    identical = 0
    for f in new_files:
        g = Path(args.reference) / f.name
        if not g.exists():
            print(f"  {f.name}: missing from the reference")
            continue
        a = pydicom.dcmread(f).pixel_array.astype(np.int64)
        b = pydicom.dcmread(g).pixel_array.astype(np.int64)
        d = np.abs(a - b)
        same = np.array_equal(a, b)
        identical += same
        result = "identical" if same else f"different, max {d.max()}, mean {d.mean():.2e}"
        print(f"  {f.name}: {result}")
    print(f"{identical}/{len(new_files)} identical")


if __name__ == "__main__":
    main()
