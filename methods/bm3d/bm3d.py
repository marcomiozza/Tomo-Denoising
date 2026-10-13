#!/usr/bin/env python3
"""
BM3D on DBT projections.

The filter is the C++ binary built from bm3d.cpp (OpenCV, xphoto module), with
the final parameters read from config.json and passed as --KEY=VALUE flags.
This script converts the DICOM projections to 16-bit TIFF, runs the binary and
writes the DICOMs back with the header of the original.

The binary is built once with setup.sh (or by hand):
    cmake -S methods/bm3d -B methods/bm3d/build -DCMAKE_BUILD_TYPE=Release
    cmake --build methods/bm3d/build

OpenCV's parallel block matching is not deterministic run-to-run: two runs on
the same projection can differ by ~10^-3 on average.

Usage:
    python methods/bm3d/bm3d.py <case|folder|file.dcm> -o <output_folder>
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.dicom_io import (find_projections, output_name, read_pixels,  # noqa: E402
                              write_dicom)

HERE = Path(__file__).resolve().parent
BINARY = HERE / "build" / "bm3d"
CONFIG = HERE / "config.json"


def config_flags(config: Path = CONFIG) -> list[str]:
    """Parameters from config.json as --KEY=VALUE flags for the binary."""
    params = json.loads(config.read_text()).get("params", {})
    flags = []
    for k, v in params.items():
        if isinstance(v, bool):
            v = "1" if v else "0"
        flags.append(f"--{k}={v}")
    return flags


def denoise(projections: list[Path], out_dir: Path, config: Path = CONFIG) -> list[Path]:
    """Denoise the projections with the BM3D binary and write the DICOMs to out_dir."""
    if not BINARY.exists():
        sys.exit(f"[ERROR] BM3D binary not found: {BINARY}. Run setup.sh to build it.")
    written = []
    with tempfile.TemporaryDirectory() as td:
        in_dir, den_dir = Path(td, "in"), Path(td, "den")
        in_dir.mkdir()
        den_dir.mkdir()
        for i, src in enumerate(projections, start=1):
            tifffile.imwrite(str(in_dir / f"proj_{i:03d}.tif"), read_pixels(src))

        cmd = [str(BINARY), str(in_dir), str(den_dir), "1", str(len(projections)),
               *config_flags(config)]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True)

        for i, src in enumerate(projections, start=1):
            den = tifffile.imread(str(den_dir / f"proj_{i:03d}.tif"))
            if den.ndim == 3:
                den = den[0]
            # The binary writes the output inverted (INVERT_16U_OUTPUT), here we
            # go back to the original scale.
            den = (65535 - den.astype(np.uint16)).astype(np.uint16)
            out = out_dir / output_name("Bm3d", src)
            write_dicom(den, src, out)
            written.append(out)
            print(f"  {src.name} -> {out.name}")
    return written


def main():
    ap = argparse.ArgumentParser(description="BM3D on DICOM projections.")
    ap.add_argument("input", help="case folder with Noisy/, folder of .dcm files or a single .dcm")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--config", default=str(CONFIG), help="JSON parameter file")
    args = ap.parse_args()

    denoise(find_projections(args.input), Path(args.out), Path(args.config))


if __name__ == "__main__":
    main()
