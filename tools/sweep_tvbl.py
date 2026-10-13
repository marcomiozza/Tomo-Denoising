#!/usr/bin/env python3
"""
Parameter sweep of TV and BL on the projections.

Applies TV and BL to the projections of a case at 3 values of each parameter
and produces 6 DICOM folders ready for FDK reconstruction. The best value is
then chosen on the reconstructed slices, with the noise and contrast metrics
used to compare the methods.

The parameter is not absolute but anchored to the noise estimated on the
projections, as k * sigma, so the k chosen on the phantom carries over
unchanged to clinical cases.

Names are short and sorted from the weakest to the strongest filtering:

    Tv4 Tv5 Tv6    increasing k on the regularization weight
    Bl1 Bl2 Bl3    increasing k on sigma_color

The folder -> parameter mapping is written to LEGEND.txt next to the folders,
so it stays readable over time.

Usage:
    python tools/sweep_tvbl.py /path/case -o <output_folder>
    python tools/sweep_tvbl.py /path/case -o <out> --only tv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.dicom_io import estimate_sigma_rel, find_projections, full_scale  # noqa: E402
from methods.tv_bl.tv_bl import BL_SIGMA_SPATIAL, denoise  # noqa: E402

# --------------------------------------------------------------------------
# Parameter grid
#
# The values are NOT absolute but multiples k of the noise estimated on the
# projections (parameter = k * sigma_rel). So the k chosen on the phantom
# carries over as is to clinical cases, whose relative noise is about ten
# times lower.
#
# TV: the first grid (k = 0.25 / 0.5 / 1) turned out too strong across the
# board. Already k = 0.25 visibly flattens the background in the
# reconstruction, even though its metrics are the best of all -- the textbook
# case where the number rewards a filter the eye rejects. The grid therefore
# goes below that value and the choice is made visually, not on CNR.
#
# BL: sigma_spatial stays fixed at 1, only the photometric selectivity moves.
# k = 1.6 is the value used so far on the phantom, k = 8 matches the
# sigma_r = 0.05 reported as optimal in previous work, applied here to the
# projections instead of the reconstructed slices.
# --------------------------------------------------------------------------
TV_GRID = [("Tv4", 0.06), ("Tv5", 0.12), ("Tv6", 0.18)]
BL_GRID = [("Bl1", 1.6), ("Bl2", 3.2), ("Bl3", 8.0)]


def write_legend(path, prefix, sigma_rel):
    """Always writes the full grid, even after a run with --only.

    Otherwise a second partial run would erase the lines of the other filter,
    whose folders are still on disk.
    """
    lines = [f"TV/BL sweep — {prefix}",
             f"estimated relative noise: sigma = {sigma_rel:.5f}",
             "the parameter of each variant is k * sigma", ""]
    for tag, k in TV_GRID:
        lines.append(f"{tag}   TV  k = {k:<5} weight = {k * sigma_rel:.5f}")
    lines.append("")
    for tag, k in BL_GRID:
        lines.append(f"{tag}   BL  k = {k:<5} sigma_color = {k * sigma_rel:.5f}"
                     f"   (sigma_spatial = {BL_SIGMA_SPATIAL})")
    lines += [
        "",
        "Each folder holds the denoised projections, ready for FDK reconstruction.",
        "Reconstruct one folder at a time and compare the slices with the same window.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"\n[LEGEND] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="case folder containing Noisy/")
    ap.add_argument("--only", choices=["tv", "bl"], help="sweep a single filter")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    args = ap.parse_args()

    case_dir = Path(args.case).resolve()
    projections = find_projections(case_dir)
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    variants = []
    if args.only != "bl":
        variants += [(tag, "tv", k) for tag, k in TV_GRID]
    if args.only != "tv":
        variants += [(tag, "bl", k) for tag, k in BL_GRID]

    print(f"[SWEEP] case {case_dir.name} — {len(projections)} projections, {len(variants)} variants")
    print(f"[SWEEP] output in {out_root}")

    sigma_rel = estimate_sigma_rel(projections, full_scale(projections[0]))
    for tag, method, k in variants:
        print(f"\n[{tag}] {method.upper()} k={k}")
        denoise(method, projections, out_root / tag, k=k, label=tag)

    write_legend(out_root / "LEGEND.txt", case_dir.name, sigma_rel)
    print(f"\n[DONE] {len(variants)} folders to reconstruct in {out_root}")


if __name__ == "__main__":
    main()
