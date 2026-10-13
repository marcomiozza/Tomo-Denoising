#!/usr/bin/env python3
"""
Single denoising pipeline for DBT projections.

Given a case folder containing a Noisy/ subfolder with the DICOM projections,
applies one or more methods (BM3D, TV, BL, U-Net) and produces the
subfolders Bm3d/, Tv/, Bl/, Unet/ next to Noisy/, with names like
Method_<prefix>_proj_NNN.dcm.

Each method is the same code as its standalone version in methods/, with the
final parameters. Methods are picked from the menu or with --methods.

    pipeline/run.sh                          # only case in pipeline/input/
    pipeline/run.sh case01                   # case in pipeline/input/
    pipeline/run.sh /path/case01 --methods bm3d,unet
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from methods.dicom_io import find_projections  # noqa: E402

INPUT_ROOT = HERE / "input"     # case folders go here (with Noisy/ inside)
OUTPUT_ROOT = HERE / "output"   # results go here, one subfolder per case

# Method order and labels. The key is also the name of the output folder.
METHODS = [
    ("Bm3d", "bm3d", "BM3D — block-matching 3D (C++ binary)"),
    ("Tv",   "tv",   "TV — Chambolle Total Variation (scikit-image)"),
    ("Bl",   "bl",   "BL — bilateral filter (scikit-image)"),
    ("Unet", "unet", "U-Net — conditioned ResUNet (torch)"),
]
FOLDER = {alias: folder for folder, alias, _ in METHODS}


def run_method(alias, projections, out_dir, ckpt=None):
    """Run one method. Imports live here so torch is loaded only for the U-Net."""
    if alias == "bm3d":
        from methods.bm3d import bm3d
        bm3d.denoise(projections, out_dir)
    elif alias in ("tv", "bl"):
        from methods.tv_bl import tv_bl
        tv_bl.denoise(alias, projections, out_dir)
    elif alias == "unet":
        from methods.unet import unet
        unet.denoise(projections, out_dir, ckpt=ckpt or unet.CKPT_DEFAULT)


def available(alias):
    """Return (ok, reason) for a method, without running it."""
    if alias == "bm3d":
        from methods.bm3d import bm3d
        if not bm3d.BINARY.exists():
            return False, f"binary not found: {bm3d.BINARY} (run setup.sh)"
    if alias == "unet":
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "torch is not installed in the environment"
    return True, ""


# --------------------------------------------------------------------------
# Case and method selection
# --------------------------------------------------------------------------
def interactive_menu():
    print("\nAvailable methods:")
    for i, (_, _, desc) in enumerate(METHODS, start=1):
        print(f"  {i}) {desc}")
    print("  a) all")
    raw = input("\nWhich methods? (e.g. '1 3' or 'a'): ").strip().lower()
    if raw in ("a", "all"):
        return [alias for _, alias, _ in METHODS]
    chosen = []
    for tok in re.split(r"[\s,]+", raw):
        if tok.isdigit() and 1 <= int(tok) <= len(METHODS):
            chosen.append(METHODS[int(tok) - 1][1])
        elif tok in FOLDER:
            chosen.append(tok)
    return [a for _, a, _ in METHODS if a in chosen]


def parse_methods(s):
    chosen = [t for t in re.split(r"[\s,]+", s.strip().lower()) if t]
    if "all" in chosen:
        return [a for _, a, _ in METHODS]
    return [a for _, a, _ in METHODS if a in chosen]


def resolve_case(arg):
    """Case folder from the command-line argument.

    - no argument: if input/ holds a single case it is used, otherwise they are listed.
    - plain name: looked up under input/.
    - existing path: used as is.
    """
    if arg:
        p = Path(arg)
        if (p / "Noisy").is_dir():
            return p.resolve()
        cand = INPUT_ROOT / arg
        if (cand / "Noisy").is_dir():
            return cand.resolve()
        sys.exit(f"[ERROR] Case not found: '{arg}'. Looked for it as a path and in {INPUT_ROOT}/")

    cases = sorted(p for p in INPUT_ROOT.iterdir() if (p / "Noisy").is_dir()) \
        if INPUT_ROOT.is_dir() else []
    if not cases:
        sys.exit(f"[ERROR] No case in {INPUT_ROOT}/ (a subfolder with Noisy/ inside is needed).")
    if len(cases) == 1:
        return cases[0].resolve()
    print(f"{INPUT_ROOT}/ contains more than one case:")
    for c in cases:
        print("   -", c.name)
    sys.exit("Pick one with: run.sh <case_name>")


def main():
    ap = argparse.ArgumentParser(description="Single denoising pipeline for DBT projections.")
    ap.add_argument("case", nargs="?", default=None,
                    help="Case name in pipeline/input/ (or a path). "
                         "If omitted and there is only one, that one is used.")
    ap.add_argument("--methods", help="Non-interactive methods: bm3d,tv,bl,unet or all")
    ap.add_argument("--ckpt", default=None, help="U-Net checkpoint (default methods/unet/checkpoints/best.pt)")
    ap.add_argument("-o", "--out", default=None,
                    help="Output folder (default pipeline/output/<case>)")
    args = ap.parse_args()

    case = resolve_case(args.case)
    noisy_dir = case / "Noisy"
    projections = find_projections(noisy_dir)
    case_dir = Path(args.out).resolve() if args.out else OUTPUT_ROOT / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"Case: {case.name}  |  projections: {len(projections)}")
    print(f"Input:  {case}")
    print(f"Output: {case_dir}")

    chosen = parse_methods(args.methods) if args.methods else interactive_menu()
    if not chosen:
        sys.exit("No method selected.")
    runnable = []
    for alias in chosen:
        ok, reason = available(alias)
        if ok:
            runnable.append(alias)
        else:
            print(f"[SKIP] {alias}: {reason}")
    if not runnable:
        sys.exit("No runnable method.")

    # The output is a complete case folder: the MAMMO reference and the Noisy
    # projections are copied next to the denoised ones.
    for mammo in case.glob("MAMMO_*.dcm"):
        shutil.copy2(mammo, case_dir / mammo.name)
    out_noisy = case_dir / "Noisy"
    if out_noisy.resolve() != noisy_dir.resolve():
        out_noisy.mkdir(exist_ok=True)
        for f in noisy_dir.glob("*.dcm"):
            shutil.copy2(f, out_noisy / f.name)

    print(f"\nRunning: {', '.join(runnable)}")
    for alias in runnable:
        print(f"\n=== {FOLDER[alias]} ===")
        run_method(alias, projections, case_dir / FOLDER[alias], args.ckpt)

    print("\n============================================================")
    print(" Done. Results in:", case_dir)
    for alias in runnable:
        print("   -", case_dir / FOLDER[alias])
    print("============================================================")


if __name__ == "__main__":
    main()
