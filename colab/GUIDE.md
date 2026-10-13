# Guide to the `Unet.ipynb` notebook — conditioned ResUNet denoiser

Notebook for **denoising DICOM images** with a residual ResUNet conditioned on the noise level (FFDNet style), meant to run on Google Colab with a GPU.

---

## 1. Goal and deployment context

The goal is to train a network that, given a noisy slice, outputs the same image "clean". The **actual deployment domain is tomosynthesis projections** (texture-rich, correlated noise). Mammograms are the training data, and the phantom is the test proxy for tomo.

Since the dataset has no real (clean, noisy) pairs, the noise is generated **synthetically** during training, with a physical model calibrated on real tomo.

---

## 2. Prerequisites

- Colab runtime **with a GPU** (a T4 is fine).
- A `MyDrive/dataset/` folder with the `.dcm` files.
- Free space on Drive for checkpoints and outputs (~1–2 GB).

The notebook creates this structure on Drive automatically:

```
MyDrive/resunet/
├── checkpoints/         # last.pt, best.pt, epoch_NNN.pt
├── logs/                # training_log.csv
├── masks/               # breast mask cache (bit-packed npz)
└── splits.json          # deterministic train/val/test split
```

---

## 3. Execution order

The first time, run the cells top to bottom. After a Colab restart it is enough to run:

1. **Cells 1–2** (imports + Drive mount)
2. **Cell 4** (CFG)
3. **Cells 5–8** (DICOM helpers, mask, split)
4. **Cells 14–22** (noise, dataset, architecture, loss, optimizer, checkpoint)
5. **Cell 26** (training, resumes from `last.pt`)

The inference/test cells (28 onwards) run after training, or at any time: they reload `best.pt` on their own.

---

## 4. Key design concepts

### 4.1 Residual learning

The network predicts an **estimate of the noise**, not the clean image:

```
clean_pred = noisy - noise_pred
```

The last 1×1 conv is **zero-initialized**: at step 0 `noise_pred = 0` → `clean_pred = noisy`. This is a safe baseline, and the network only learns the deviation from the pass-through.

### 4.2 Tomo-NPS noise model (how we got there)

**Starting point.** Earlier versions used white Poisson+Gaussian noise. When tested on tomo/phantom the network over-smoothed, because the grain of tomo noise is visibly different from mammo noise: tomo has a **finer grain, more uniform in frequency** (nearly flat or slightly rising NPS), while in mammo the low frequencies dominate.

**NPS measurement.** The signature of the grain is its **radial power spectral density (NPS)**. The procedure (implemented in `tools/tomo_noise.py`):
1. Take a flat ROI on the tomo (uniform phantom background, non-saturated area).
2. Compute the squared magnitude of the FFT and average it over radial rings normalized to `r ∈ [0, 1]` (Nyquist).
3. The resulting profile (48 points) is `NPS_RADIAL`, embedded in cell 14 of the notebook.

**`TomoNPSNoise` model.** Given a clean patch, it produces noise colored with that NPS:
1. Generate white Gaussian noise.
2. Multiply in the frequency domain by `√NPS_RADIAL` (coloring filter via FFT).
3. Normalize by the standard deviation.
4. Scale by a **signal-dependent** amplitude:
   ```
   sigma(clean) = sqrt(read² + level² * (1 - clean))
   ```
   Here `1 - clean` is the photon signal (after MONOCHROME1 inversion): the darker the pixel (high radiological density = few photons), the more noise. This is the real physics of Poisson shot noise.

`level` is sampled uniformly in `[TOMO_LEVEL_MIN, TOMO_LEVEL_MAX]` (default 0.002–0.020) at each batch, so the network learns to denoise at **different strengths** (essential for conditioning).

### 4.3 FFDNet-style conditioning (in_ch=2)

The network takes **2 input channels**: `[noisy_image, sigma_channel]`.

The sigma channel is a constant map = `level × SIGMA_GAIN` (default `SIGMA_GAIN=50`, so level ≈ 0.01 → sigma ≈ 0.5, salient for the network).

`make_input(noisy, level)` builds this input both in training (level = `(B,1,1,1)` tensor sampled from the noise) and at inference (level = float chosen by the user).

**`CFG.INFER_LEVEL` is the strength knob at inference**, tunable on tomo without retraining:
- high (e.g. 0.015–0.020) → aggressive denoising, risk of oversmoothing
- low (e.g. 0.003–0.005) → gentle, preserves texture

The reference results are produced with `INFER_LEVEL = 0.003`, which is the default of the notebook and of `methods/unet/unet.py`.

This solved the oversmoothing problem cleanly: instead of fighting with the loss (gradient_l1, texture_loss, both failed), the denoising strength is simply an inference parameter.

### 4.4 Loss and metrics

**Training loss:** `masked_charbonnier(noise_pred, noise_true)`, a Charbonnier loss masked on the breast, with no extra terms.

Conditioning handles "how much to denoise", so a gradient/texture term in the loss is no longer needed (it created a fidelity↔texture tug of war that led nowhere).

**Diagnostics (not objectives):**
- `masked_hf_ratio` (`hf=` in the logs): ratio of high-frequency power denoised/clean on the breast. `hf < 1` = oversmoothing, `hf > 1` = residual noise, `hf ≈ 0.9–1.0` is optimal. It is expected below 1 because the clean image already has its own detector grain.
- `masked_ssim`: SSIM computed on the breast bounding box (not on the whole patch).

**Best checkpoint selection:** on `val_psnr`. On the synthetic task PSNR does not reward smoothing (flattening moves away from the clean target), so it is an honest proxy.

### 4.5 Breast mask

Otsu + largest connected component + morphological closing, cached on disk (bit-packed npz). Used for:
- **Smart cropping**: discards patches with less than `MIN_BREAST_FRACTION` (30%) tissue.
- **Masked loss**: the background does not contribute to the gradient.
- **Metrics**: PSNR/SSIM/hf computed on the breast only. **Not comparable with full-image PSNR**.

### 4.6 Conditioned ResUNet architecture

5-level ResUNet, `in_ch=2` (image + sigma channel), `BASE_CHANNELS=48`:

- `ResBlock`: `Conv → GroupNorm → SiLU → Conv → GroupNorm → 1×1 skip`.
- Downsampling with `AvgPool2d` (less aliasing than MaxPool on noisy signals).
- Upsampling with `ConvTranspose2d` + `bilinear` when sizes do not match.
- `head`: zero-initialized 1×1 conv.

### 4.7 Patch training + sliding-window inference

Training on **256×256** patches with smart cropping on the breast. Inference on the full image (~3580×2812) via a Gaussian sliding window: patches overlap by `PATCH_SIZE/4` and contributions are averaged with a 2D Gaussian to remove seams.

### 4.8 Colab resilience

- Data on the local disk `/content/dataset/` (no Drive FUSE during training).
- Deterministic split in `splits.json`.
- Atomic checkpoints (`os.replace`). `last.pt` saved every epoch, `best.pt` when `val_psnr` improves.
- Auto-resume from `last.pt` if `CFG.RESUME = True`.

---

## 5. Cell by cell (summary)

| # | Title | What it does |
|---|---|---|
| 2 | Imports | Dependencies, seed, device |
| 4 | `CFG` configuration | All parameters |
| 5–6 | Dataset copy | From Drive to `/content/dataset/` |
| 8 | DICOM helpers | `load_dicom_normalized`, `save_dicom_from_norm` |
| 10 | Breast mask | `compute_breast_mask`, cache |
| 12 | Split | `splits.json` |
| 14 | `TomoNPSNoise` noise | NPS model + `make_input` |
| 16 | Dataset/loader | `MammoDenoiseDataset` |
| 18 | ResUNet | Architecture |
| 20 | Loss/metrics | Charbonnier + hf/ssim diagnostics |
| 22 | Optimizer | AdamW + warmup + cosine |
| 24 | Checkpoint | Save/load, auto-resume |
| 26 | **Training** | Main loop |
| 28 | Sliding window | `sliding_window_inference` |
| 30 | Test set | Synthetic metrics + DICOM output |
| 32 | Single DICOM denoise | Tune `INFER_LEVEL` |
| 40 | `INFER_LEVEL` sweep | Strength calibration on the phantom |
| 42 | Denoise 11 tomo projections | Output for reconstruction |

---

## 6. Reading the results

### During training

Each epoch prints:
```
[VAL] ep 012/120 | loss 0.00712 | PSNR 33.87 | ssim 0.891 | hf 0.94 | lr 1.8e-4
```

- **`val_psnr`**: rises steadily, then plateaus. Used for best checkpoint selection.
- **`hf`**: target 0.9–1.0. Below 0.7 = oversmoothing, above 1.1 = residual noise.
- **`val_ssim`**: indicative, > 0.85 is already good.

### Inference on tomo/phantom

Use the `INFER_LEVEL` sweep cell (cell 40) to find the best strength visually on the phantom. Then set that value in `CFG.INFER_LEVEL` for the 11 projections (cell 42).

Locally (from the repository root): `.venv/bin/python methods/unet/unet.py <projections_folder> -o <output_folder>` (level 0.003 by default)

---

## 7. Tuning

| Parameter | Effect | When to change it |
|---|---|---|
| `INFER_LEVEL` | Denoising strength at inference | **First lever**: tune it on tomo without retraining |
| `TOMO_LEVEL_MIN/MAX` | Strength range in training | To cover different strengths |
| `BASE_CHANNELS` | Model capacity | PSNR plateaus early → increase |
| `MIN_BREAST_FRACTION` | Crop selection | Wrong masks → lower it |
| `N_CROPS_TRAIN` | Crops/image/epoch | Epochs too fast → increase |
| `LR` | Learning rate | Loss oscillates → lower it |

---

## 8. Common problems

**Training restarts from scratch**
Check that `CFG.RESUME = True` and that `last.pt` is in `CFG.CKPT_DIR`.

**`hf` drops quickly below 0.7**
Oversmoothing in training. Lower `INFER_LEVEL` at inference (it does not change training). If the problem is already there in training, the `TOMO_LEVEL_MIN/MAX` range may be too high.

**`val_psnr` does not rise**
Mask too restrictive (`mask.sum() ≈ 0`) or synthetic noise with zero variance. Check cell 14.

**GPU OOM**
Lower `BATCH_SIZE` (4) or `PATCH_SIZE` (192).

---

## 9. Conventions

- Output is always **DICOM** (never PNG/JPG). `matplotlib imshow` is only for inline display.
- PSNR/SSIM/hf are always **masked on the breast**, not comparable with full-image metrics.
