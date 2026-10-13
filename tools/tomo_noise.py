#!/usr/bin/env python3
"""
tomo_noise.py — Generator of "tomosynthesis-like" noise to add to mammograms.

Idea (from the analysis on the phantom):
  A tomo projection is the same image at ~1/8 of the dose, on the same detector.
  The noise is not arbitrary: it has two signatures measurable on real data
    1) signal-dependent AMPLITUDE  -> var(S) = a*(S - offset)   (Poisson)
    2) GRAIN (texture)             -> spectral shape, the NPS of the tomo
  Both are CALIBRATED on a reference tomo projection (no mammo-tomo
  subtraction, no registration needed, unlimited realizations).

The NPS_RADIAL profile used in training (colab/Unet.ipynb) is the grain
measured here on the phantom's Tomo_6.dcm, resampled on 48 points. It can be
rebuilt with --nps.

Usage:
  python3 tomo_noise.py --mammo MAMMO.dcm --ref PROJ06.dcm --out OUT.dcm [--strength 1.3]
  python3 tomo_noise.py --nps --ref Tomo_6.dcm      # prints NPS_RADIAL for the notebook
"""
import argparse
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter, uniform_filter


def load_dcm(path):
    ds = pydicom.dcmread(path)
    return ds, ds.pixel_array.astype(np.float64)


def _flat_rois(img, win=32, step=32, lo=None, hi=None):
    """Return (mean_signal, noise_variance) for the flattest ROIs, per signal bin."""
    if lo is None:
        lo = np.percentile(img, 2)
    if hi is None:
        hi = np.percentile(img, 99) * 0.98  # avoid saturation
    hp = img - gaussian_filter(img, 4)
    gmag = np.hypot(*np.gradient(gaussian_filter(img, 8)))
    H, W = img.shape
    rec = []
    for y in range(0, H - win, step):
        for x in range(0, W - win, step):
            m = img[y:y + win, x:x + win].mean()
            if m < lo or m > hi:
                continue
            h = hp[y:y + win, x:x + win]
            mad = 1.4826 * np.median(np.abs(h - np.median(h)))
            rec.append((m, mad * mad, gmag[y:y + win, x:x + win].mean()))
    return np.array(rec)


def estimate_noise_model(tomo, win=32):
    """Calibrate (a, offset) of the law var=a*(S-offset) and the NPS grain from the tomo."""
    # AMPLITUDE: simple, robust calibration on a flat, non-saturated ROI of the tomo.
    # offset = detector pedestal (dark), estimated as the 1st percentile; var = a*(S-offset).
    offset = float(np.percentile(tomo, 1))
    s_star = float(np.percentile(tomo[tomo < np.percentile(tomo, 99) * 0.98], 75))
    sig_star = noise_std_at(tomo, s_star)
    a = sig_star ** 2 / max(s_star - offset, 1.0)

    # GRAIN: radial NPS on a flat ROI at mid, non-saturated signal
    yc, xc = _best_flat_center(tomo, s_star, win=256)
    patch = tomo[yc - 128:yc + 128, xc - 128:xc + 128]
    hp = patch - gaussian_filter(patch, 4)
    F = np.fft.fftshift(np.abs(np.fft.fft2(hp)) ** 2)
    cy, cx = np.array(F.shape) // 2
    Y, X = np.indices(F.shape)
    R = np.hypot(Y - cy, X - cx).astype(int)
    prof = np.bincount(R.ravel(), F.ravel()) / np.maximum(np.bincount(R.ravel()), 1)
    prof /= prof[1:].mean()
    fr_axis = np.arange(len(prof)) / (patch.shape[0] / 2)
    return dict(a=a, offset=offset, prof=prof, fr_axis=fr_axis,
                s_star=s_star, sig_star=sig_star)


def noise_std_at(img, target, win=48, tol=0.15):
    """Robust noise estimate at signal 'target'.

    Structure can only INFLATE the local variance, never lower it: so among many
    windows close to the target signal the 5th percentile of the local stds is
    taken (= the flattest window), immune to the phantom structure.
    """
    hp = img - gaussian_filter(img, 4)
    H, W = img.shape
    stds = []
    for y in range(0, H - win, win):
        for x in range(0, W - win, win):
            m = img[y:y + win, x:x + win].mean()
            if abs(m - target) > tol * target:
                continue
            h = hp[y:y + win, x:x + win]
            stds.append(1.4826 * np.median(np.abs(h - np.median(h))))
    if not stds:
        return 0.0
    return float(np.percentile(stds, 5))


def _best_flat_center(img, target_signal, win=256):
    g = np.hypot(*np.gradient(gaussian_filter(img, 8)))
    near = np.abs(img - target_signal) < 0.15 * target_signal
    score = uniform_filter((near & (g < np.quantile(g[near], 0.3))).astype(float), win // 4)
    yc, xc = np.unravel_index(np.argmax(score), score.shape)
    H, W = img.shape
    return min(max(yc, win // 2), H - win // 2), min(max(xc, win // 2), W - win // 2)


def colored_noise(shape, model, rng):
    """White noise colored with the spectral shape (NPS) of the tomo -> same grain."""
    H, W = shape
    FR = np.hypot(*np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')) * 2
    A = np.interp(FR.ravel(), model['fr_axis'],
                  np.sqrt(np.clip(model['prof'], 0, None))).reshape(shape)
    out = np.fft.ifft2(np.fft.fft2(rng.standard_normal((H, W))) * A).real
    return out / out.std()


def radial_nps(model, n=48):
    """NPS profile resampled on n points between frequency 0 and Nyquist.

    This is the NPS_RADIAL format in the notebook. The profile is renormalized so
    that the mean of the points after the first equals 1, like the script's one.
    """
    v = np.interp(np.linspace(0, 1, n), model['fr_axis'], model['prof'])
    return v / v[1:].mean()


def tomize(mammo, tomo_model, strength=1.2, seed=0):
    """Add to the mammogram noise with the grain and amplitude-vs-signal of the tomo.

    sigma(S) = sqrt(a*(S - offset)) * strength   (Poisson, signal-dependent)
    The noise (colored with the tomo NPS = same grain) is added on top of the
    mammogram. 'strength' scales the overall amount: the default 1.2 gives a
    result slightly noisier than the nominal tomo.
    """
    rng = np.random.default_rng(seed)
    n = colored_noise(mammo.shape, tomo_model, rng)
    sigma = np.sqrt(np.clip(tomo_model['a'] * (mammo - tomo_model['offset']), 0, None)) * strength
    return mammo + n * sigma


def save_dcm(arr, src_ds, path):
    """Clone the source metadata, keep MONOCHROME1/bit depth."""
    ds = pydicom.dcmread(src_ds.filename)
    maxv = (1 << ds.BitsStored) - 1
    ds.PixelData = np.clip(arr, 0, maxv).astype(np.uint16).tobytes()
    ds.save_as(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mammo')
    ap.add_argument('--ref', required=True, help='reference tomo projection (e.g. proj06)')
    ap.add_argument('--out')
    ap.add_argument('--nps', action='store_true',
                    help='only print the NPS_RADIAL profile (48 points) measured on --ref')
    ap.add_argument('--strength', type=float, default=1.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    _, ref = load_dcm(args.ref)
    tomo_model = estimate_noise_model(ref)
    if args.nps:
        v = radial_nps(tomo_model)
        rows = [", ".join(f"{x:.3f}" for x in v[i:i + 10]) for i in range(0, len(v), 10)]
        print("NPS_RADIAL = [" + ",\n              ".join(rows) + "]")
        return
    if not (args.mammo and args.out):
        ap.error("--mammo and --out are required (or --nps)")
    mg_ds, mg = load_dcm(args.mammo)
    print(f"tomo model: a={tomo_model['a']:.3f} offset={tomo_model['offset']:.0f} "
          f"(@signal~{tomo_model['s_star']:.0f} std_tomo={tomo_model['sig_star']:.1f})")
    out = np.clip(tomize(mg, tomo_model, strength=args.strength, seed=args.seed), 0, None)
    save_dcm(out, mg_ds, args.out)
    print(f"saved {args.out}  (strength={args.strength})")


if __name__ == '__main__':
    main()
