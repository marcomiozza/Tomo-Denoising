#!/usr/bin/env python3
"""
Denoising of DBT projections with the noise-conditioned ResUNet (FFDNet style).

Reproduces the inference of the training notebook (colab/Unet.ipynb): the
network estimates the noise and subtracts it (clean = noisy - noise_pred),
takes two input channels [image, sigma channel] and runs as a sliding window
with Gaussian weighting. Pixels are normalized on the fixed detector full
scale (2**BitsStored - 1), not on the min/max of each image.

The denoising strength is the level parameter. The reference results are
produced with level = 0.003 and the checkpoint checkpoints/best.pt (epoch 55).

Runs on GPU if available, otherwise on CPU (about 45 s per projection).

Usage:
    python methods/unet/unet.py <case|folder|file.dcm> -o <output_folder>
    python methods/unet/unet.py <input> -o <out> --level 0.003 --ckpt other.pt
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.dicom_io import find_projections, output_name  # noqa: E402

CKPT_DEFAULT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"


# ---------------------------------------------------------------------------
# Configuration (subset of the notebook CFG relevant to inference)
# ---------------------------------------------------------------------------
class CFG:
    PATCH_SIZE    = 256
    BASE_CHANNELS = 48
    SIGMA_GAIN    = 50.0      # sigma channel gain (same value as in training)
    INFER_LEVEL   = 0.003     # denoising strength used for the reference results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"   # autocast only on CUDA, off on CPU


# ---------------------------------------------------------------------------
# DICOM helpers (normalized load/save, FIXED pmax from BitsStored)
# ---------------------------------------------------------------------------
def get_pixel_max(ds):
    return float((1 << int(getattr(ds, "BitsStored", 16))) - 1)


def _dcmread_robust(path, max_retries=4):
    for attempt in range(max_retries):
        try:
            return pydicom.dcmread(path)
        except OSError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2.0 * (2 ** attempt)
            print(f"  OSError on {os.path.basename(path)}: {e} "
                  f"— retrying in {wait:.0f}s ({attempt+1}/{max_retries})")
            time.sleep(wait)


def load_dicom_normalized(path):
    ds   = _dcmread_robust(path)
    arr  = ds.pixel_array.astype(np.float32)
    pmax = get_pixel_max(ds)
    mono1 = (getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")
    if mono1:
        arr = pmax - arr
    arr = np.clip(arr / pmax, 0.0, 1.0)
    return arr, {"pmax": pmax, "mono1": mono1, "shape": arr.shape, "path": path}


def save_dicom_from_norm(arr_norm, source_path, out_path):
    """Save arr_norm [0,1] as DICOM, cloning the metadata of the source.
    Rescales with the FIXED pmax (bit depth), NOT per-image min/max."""
    ds_src = _dcmread_robust(source_path)
    ds_out = copy.deepcopy(ds_src)
    pmax   = get_pixel_max(ds_src)
    mono1  = (getattr(ds_src, "PhotometricInterpretation", "") == "MONOCHROME1")

    arr_int = np.clip(np.round(arr_norm * pmax), 0, pmax).astype(np.uint16)
    if mono1:
        arr_int = (pmax - arr_int.astype(np.float32)).astype(np.uint16)

    ds_out.PixelData            = arr_int.tobytes()
    ds_out.Rows, ds_out.Columns = arr_int.shape
    ds_out.BitsAllocated        = 16
    ds_out.BitsStored           = int(getattr(ds_src, "BitsStored", 16))
    ds_out.HighBit              = ds_out.BitsStored - 1
    ds_out.PixelRepresentation  = 0
    ds_out.SamplesPerPixel      = 1
    ds_out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    try:
        ds_out.save_as(out_path, enforce_file_format=True)
    except TypeError:
        ds_out.save_as(out_path, write_like_original=False)


# ---------------------------------------------------------------------------
# Noise conditioning: 2-channel input [image, sigma channel]
# ---------------------------------------------------------------------------
def make_input(noisy, level):
    """2-channel input: [noisy_image, sigma_channel].
    The sigma channel is constant = level * SIGMA_GAIN."""
    if not torch.is_tensor(level):
        level = torch.full((noisy.shape[0], 1, 1, 1), float(level),
                           device=noisy.device, dtype=noisy.dtype)
    sig = (level * CFG.SIGMA_GAIN).expand(noisy.shape[0], 1, noisy.shape[-2], noisy.shape[-1])
    return torch.cat([noisy, sig], dim=1)


# ---------------------------------------------------------------------------
# Architecture — conditioned ResUNet (in_ch=2)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = min(8, out_ch)
        while out_ch % g != 0:
            g //= 2
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool  = nn.AvgPool2d(2)
        self.block = ResBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ResBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=48):
        super().__init__()
        b = base
        self.enc0 = ResBlock(in_ch, b)
        self.enc1 = Down(b,    b*2)
        self.enc2 = Down(b*2,  b*4)
        self.enc3 = Down(b*4,  b*8)
        self.enc4 = Down(b*8,  b*8)
        self.dec3 = Up(b*8,  b*8, b*4)
        self.dec2 = Up(b*4,  b*4, b*2)
        self.dec1 = Up(b*2,  b*2, b)
        self.dec0 = Up(b,    b,   b)
        self.head = nn.Conv2d(b, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d  = self.dec3(e4, e3)
        d  = self.dec2(d,  e2)
        d  = self.dec1(d,  e1)
        d  = self.dec0(d,  e0)
        return self.head(d)


def load_ckpt(path, model):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return ck.get("epoch", -1), ck.get("best_val", 0.0)


# ---------------------------------------------------------------------------
# Sliding-window inference (conditioned) — clean_pred = noisy - noise_pred
# ---------------------------------------------------------------------------
def sliding_window_inference(model, image_t, level, patch=256, overlap=64):
    """image_t: (1,1,H,W) on device; level: denoising strength (float) → clean_pred."""
    model.eval()
    _, _, H, W = image_t.shape
    step  = patch - overlap
    pad_h = (step - max(H - patch, 0) % step) % step if H >= patch else patch - H
    pad_w = (step - max(W - patch, 0) % step) % step if W >= patch else patch - W
    xp    = F.pad(image_t, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = xp.shape[-2:]

    out = torch.zeros_like(xp)
    wts = torch.zeros_like(xp)

    sigma  = patch / 6.0
    coords = torch.arange(patch, dtype=torch.float32, device=image_t.device) - patch / 2.0
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    win    = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    ys = list(range(0, Hp - patch + 1, step))
    xs = list(range(0, Wp - patch + 1, step))
    if not ys or ys[-1] != Hp - patch: ys.append(Hp - patch)
    if not xs or xs[-1] != Wp - patch: xs.append(Wp - patch)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                pin = xp[:, :, y:y+patch, x:x+patch]
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    cp = (pin - model(make_input(pin, level))).float()
                out[:, :, y:y+patch, x:x+patch] += cp * win
                wts[:, :, y:y+patch, x:x+patch] += win

    return (out / wts.clamp(min=1e-6))[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Denoising of one acquisition
# ---------------------------------------------------------------------------
def load_model(ckpt=CKPT_DEFAULT):
    ckpt = Path(ckpt)
    if not ckpt.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt}")
    model = ResUNet(in_ch=2, out_ch=1, base=CFG.BASE_CHANNELS).to(device)
    ep, best_val = load_ckpt(str(ckpt), model)
    model.eval()
    print(f"[Unet] checkpoint {ckpt.name} | epoch={ep} | best_psnr={best_val:.4f} | "
          f"device {device}")
    return model


def denoise(projections, out_dir: Path, level: float = CFG.INFER_LEVEL,
            ckpt=CKPT_DEFAULT) -> list[Path]:
    """Denoise the projections and write the DICOMs to out_dir."""
    model = load_model(ckpt)
    print(f"[Unet] level = {level} | sigma_gain = {CFG.SIGMA_GAIN} | "
          f"projections: {len(projections)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in projections:
        t0 = time.time()
        arr, _ = load_dicom_normalized(str(p))
        input_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)
        den_t = sliding_window_inference(
            model, input_t, level,
            patch=CFG.PATCH_SIZE, overlap=CFG.PATCH_SIZE // 4
        ).clamp(0, 1)
        out = out_dir / output_name("Unet", Path(p))
        save_dicom_from_norm(den_t.squeeze().detach().cpu().numpy(), str(p), str(out))
        written.append(out)
        print(f"  {Path(p).name} -> {out.name}  ({time.time()-t0:.1f}s)")
    return written


def main():
    ap = argparse.ArgumentParser(description="Denoise DICOM projections with the conditioned ResUNet.")
    ap.add_argument("input", help="case folder with Noisy/, folder of .dcm files or a single .dcm")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--level", type=float, default=CFG.INFER_LEVEL,
                    help=f"denoising strength (default {CFG.INFER_LEVEL}, reference value)")
    ap.add_argument("--ckpt", default=str(CKPT_DEFAULT),
                    help="checkpoint .pt (default checkpoints/best.pt)")
    args = ap.parse_args()

    denoise(find_projections(args.input), Path(args.out), args.level, args.ckpt)


if __name__ == "__main__":
    main()
