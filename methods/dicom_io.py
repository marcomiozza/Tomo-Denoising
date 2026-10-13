"""
Reading and writing of DICOM projections, shared by all methods.

Every method (BM3D, TV, BL, U-Net) and the pipeline go through here, so the
standalone version of a method and the one called by the pipeline produce
the same files.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid


def _natural_key(name):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def find_projections(source) -> list[Path]:
    """Return the DICOM files to process, sorted.

    `source` can be a .dcm file, a folder of .dcm files or a case folder
    containing Noisy/. Files are sorted by InstanceNumber, falling back to
    natural filename order when the tag is missing or repeated.
    """
    p = Path(source)
    if p.is_file():
        return [p]
    if (p / "Noisy").is_dir():
        p = p / "Noisy"
    files = list(p.glob("*.dcm"))
    if not files:
        sys.exit(f"[ERROR] No .dcm files in {p}")

    def instance_key(f):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    keys = [instance_key(f) for f in files]
    if len(set(keys)) < len(keys) or all(k == 0 for k in keys):
        return sorted(files, key=lambda f: _natural_key(f.name))
    return sorted(files, key=instance_key)


def output_name(method: str, source: Path) -> str:
    """Name of the denoised DICOM, e.g. case01_R-MLO_Noisy_proj_006 -> Tv_case01_R-MLO_proj_006.dcm."""
    return f"{method}_{source.stem.replace('_Noisy', '')}.dcm"


def read_pixels(path: Path) -> np.ndarray:
    """Raw projection pixels as uint16, with no transformation."""
    return pydicom.dcmread(str(path)).pixel_array.astype(np.uint16)


def full_scale(path: Path) -> float:
    """Full-scale pixel value, 2**BitsStored - 1.

    The phantom is 16 bit (65535), clinical acquisitions are 13 bit (8191).
    """
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return float(2 ** int(getattr(ds, "BitsStored", 16)) - 1)


def estimate_sigma_rel(projections: list[Path], norm: float) -> float:
    """Noise relative to full scale, estimated inside the breast on the
    central projection.

    Uses the difference between adjacent rows divided by sqrt(2): at that
    distance the anatomical signal is almost identical and mostly noise is
    left. Air pixels, which in MONOCHROME1 projections are the ones saturated
    at full scale, are discarded, and a row pair is kept only if both pixels
    fall inside the breast.

    The median absolute deviation of the differences is rescaled to a standard
    deviation. Anatomical edges produce differences much larger than the
    noise: they are few but weigh heavily on a second moment, while they
    barely affect a median statistic.

    This anchors the TV and BL parameters to the noise. The phantom has a
    relative noise of ~0.04, clinical acquisitions ~0.003, so the same
    absolute parameter would act very differently. Written as k * sigma_rel,
    a single coefficient k holds for all cases.
    """
    mid = projections[len(projections) // 2]
    a = pydicom.dcmread(str(mid)).pixel_array.astype(np.float64) / norm
    breast = a < 0.995 * float(a.max())
    d = (a[1:, :] - a[:-1, :])[breast[1:, :] & breast[:-1, :]]
    if d.size < 1000:
        sys.exit("[ERROR] not enough tissue to estimate the noise on "
                 f"projection {mid.name}")
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / np.sqrt(2)


def write_dicom(pixels: np.ndarray, template: Path, out_path: Path):
    """Write the denoised pixels by cloning the header of the original projection.

    MONOCHROME1, 16 bit and a new SOPInstanceUID, ready for FDK reconstruction.
    BM3D, TV and BL all write through here.
    """
    ds = pydicom.dcmread(str(template))
    img = pixels.astype(np.uint16)
    rows, cols = img.shape

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)
