#!/usr/bin/env python3
"""
Total Variation (Chambolle) and bilateral filter on DBT projections.

The parameters are the final ones, expressed as multiples of the relative
noise estimated on the acquisition (parameter = k * sigma_rel, see
dicom_io.estimate_sigma_rel), so the same coefficients hold on the phantom
and on clinical cases. The coefficients come from the phantom sweep
(tools/sweep_tvbl.py), chosen visually on the FDK reconstructions.

    TV  weight      = 0.12 * sigma_rel   (variant Tv5), max 40 iterations
    BL  sigma_color = 3.2  * sigma_rel   (variant Bl2), sigma_spatial = 1

Projections are normalized to [0,1] with respect to the detector full scale
(2**BitsStored - 1), not 65535, because weight and sigma_color are absolute
thresholds in that space.

Usage:
    python methods/tv_bl/tv_bl.py tv <case|folder|file.dcm> -o <output_folder>
    python methods/tv_bl/tv_bl.py bl <case|folder|file.dcm> -o <output_folder>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.dicom_io import (estimate_sigma_rel, find_projections,  # noqa: E402
                              full_scale, output_name, read_pixels, write_dicom)

TV_K = 0.12            # Tv5
TV_MAX_ITER = 40
BL_K = 3.2             # Bl2
BL_SIGMA_SPATIAL = 1

FOLDER = {"tv": "Tv", "bl": "Bl"}


def apply_filter(img: np.ndarray, method: str, param: float, norm: float) -> np.ndarray:
    """Apply TV or BL to a uint16 projection and return uint16 on the original scale.

    `param` is the TV weight or the bilateral sigma_color, already multiplied
    by sigma_rel.
    """
    img_f = img.astype(np.float32) / norm
    if method == "tv":
        den = denoise_tv_chambolle(img_f, weight=param, max_num_iter=TV_MAX_ITER,
                                   channel_axis=None)
    else:
        den = denoise_bilateral(img_f, sigma_color=param, sigma_spatial=BL_SIGMA_SPATIAL,
                                channel_axis=None)
    return np.clip(den * norm, 0, 65535).astype(np.uint16)


def denoise(method: str, projections: list[Path], out_dir: Path, k: float | None = None,
            label: str | None = None) -> list[Path]:
    """Denoise all projections of an acquisition and write the DICOMs to out_dir.

    sigma_rel is estimated once on the central projection and used for all of
    them. `k` overrides the default coefficient (used by the sweep), `label`
    the prefix of the output names.
    """
    if k is None:
        k = TV_K if method == "tv" else BL_K
    label = label or FOLDER[method]
    norm = full_scale(projections[0])
    sigma_rel = estimate_sigma_rel(projections, norm)
    # Python float, not np.float64: with numpy 2 an np.float64 scalar pushes
    # the filter computation to double precision, while the reference results
    # were produced in float32.
    param = float(k * sigma_rel)
    param_name = "weight" if method == "tv" else "sigma_color"
    print(f"[{label}] full scale {norm:.0f}, sigma_rel {sigma_rel:.5f}, "
          f"{param_name} = {k} * sigma_rel = {param:.6f}")

    written = []
    for src in projections:
        den = apply_filter(read_pixels(src), method, param, norm)
        out = out_dir / output_name(label, src)
        write_dicom(den, src, out)
        written.append(out)
        print(f"  {src.name} -> {out.name}")
    return written


def main():
    ap = argparse.ArgumentParser(description="TV or bilateral filter on DICOM projections.")
    ap.add_argument("method", choices=["tv", "bl"])
    ap.add_argument("input", help="case folder with Noisy/, folder of .dcm files or a single .dcm")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--k", type=float, default=None,
                    help=f"noise coefficient (default TV {TV_K}, BL {BL_K})")
    args = ap.parse_args()

    projections = find_projections(args.input)
    denoise(args.method, projections, Path(args.out), k=args.k)


if __name__ == "__main__":
    main()
