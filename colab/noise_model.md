# Tomo-like noise model

Tomo and mammo use the same detector, but tomo has ~1/8 of the dose. The noise differs in two ways:

- **Amplitude:** similar at equal signal (both Poisson), but tomo has more of it because there are fewer photons
- **Grain:** tomo noise is much whiter and finer than mammo noise, and this is the visual signature that matters

## The two components

### Amplitude (Poisson)

```
var(S) = a * (S - offset)
```

`offset` is the detector pedestal (1st percentile of the image). `a` is calibrated on the real tomo: flat ROIs at a known signal `S*` are located, the noise is measured there (MAD on a high-pass), and `a = sigma²/(S*−offset)`. To stay conservative the 5th percentile of the stds across candidate windows is taken, since anatomical structure can only inflate the local variance, never lower it.

### Grain (radial NPS)

A 256×256 patch is taken on a flat area of the tomo, a slow blur (σ=4px) is subtracted to remove the anatomical structure, and a 2D FFT is computed. The power spectrum `|FFT|²` reduced to a radial profile (mean per ring) is the NPS. Normalized to mean=1, it describes the shape of the grain.

## Generation

1. Sample white Gaussian noise (std=1)
2. Color it in frequency: FFT → multiply by `sqrt(NPS[r])` → IFFT → renormalize
3. Scale pixel by pixel: `n_final = n_colored * sqrt(a*(S−offset)) * strength`
4. Add it to the mammogram

`strength` (default ~1.35) is the only knob. The result is slightly noisier than the nominal tomo.

## Two implementations

**`tools/tomo_noise.py`** calibrates `a`, `offset` and the NPS at runtime from `--ref` (a real tomo projection, typically proj06). Output is DICOM.

```
python3 tools/tomo_noise.py --mammo M.dcm --ref PROJ06.dcm --out O.dcm --strength 1.35
```

**`TomoNPSNoise` in the notebook** follows the same principle, but with a precomputed embedded NPS (48 values from the phantom's Tomo_6). The amplitude formula is adapted to the normalized [0,1] domain after MONOCHROME1 inversion:

```
sigma(clean) = sqrt(read² + level² * (1 − clean))
```

`(1−clean)` is the photon signal: more noise in the background (air), less in dense tissue, which is physically correct.

## What was discarded

- **Mammo−tomo subtraction:** the images are not aligned pixel by pixel
- **White Gaussian noise:** wrong grain, visually obvious
- **Subtraction in quadrature** `sqrt(σ_tomo²−σ_mammo²)`: it made the result worse, because the σ estimates are not comparable between the two images (different structures). The simple "add on top" version works better

## Validation

Visual only, in Fiji: triplets of original mammo / tomo-ized mammo / real tomo with the same brightness and contrast. Grain (fineness) and background amplitude are checked. Numbers (PSNR, SSIM) make no sense here, because there is no pixel-by-pixel ground truth.
