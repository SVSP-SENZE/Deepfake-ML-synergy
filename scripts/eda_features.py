"""EDA step 1: one row of hand-crafted measurements per training image.

Reads the original JPEG files (for file-level properties) and the decoded 224x224 cache (for pixel statistics).
Training images only; the test set is never read.
Output: experiments/eda/image_stats.csv
"""
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.common import ROOT
from src.data import train_images, with_folds

OUT = ROOT / "experiments" / "eda"

# radial frequency mask for a 224x224 spectrum, radius as a fraction of Nyquist
_fy, _fx = np.meshgrid(np.fft.fftfreq(224), np.fft.fftfreq(224), indexing="ij")
RADIUS = np.sqrt(_fx ** 2 + _fy ** 2) / 0.5


def file_props(path):
    im = Image.open(path)
    q = getattr(im, "quantization", None) or {}
    qhash = hashlib.md5(str(sorted((k, tuple(v)) for k, v in q.items())).encode()).hexdigest()[:8]
    return dict(width=im.size[0], height=im.size[1], mode=im.mode, fmt=im.format,
                file_kb=os.path.getsize(path) / 1024, q_dc=q[0][0] if q else -1, qtable=qhash,
                has_exif=bool(im.info.get("exif")), progressive=bool(im.info.get("progressive")),
                subsampling=getattr(im, "layers", -1))


def pixel_props(rgb):
    f = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256) / gray.size
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    resid = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    spec = np.abs(np.fft.fft2(gray - gray.mean())) ** 2
    dx = np.abs(np.diff(gray, axis=1))
    # 8x8 block std: low std over a large connected area = flat patch (occlusion)
    blk = gray.reshape(28, 8, 28, 8).std(axis=(1, 3))
    return dict(
        brightness=gray.mean(), contrast=gray.std(),
        saturation=hsv[..., 1].mean(),
        colourfulness=np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2),
        chroma_spread=np.abs(f - f.mean(axis=2, keepdims=True)).mean(),
        mean_r=r.mean(), mean_g=g.mean(), mean_b=b.mean(),
        entropy=-(hist[hist > 0] * np.log2(hist[hist > 0])).sum(),
        clip_dark=(gray <= 5).mean(), clip_bright=(gray >= 250).mean(),
        sharpness=np.log10(lap.var() + 1e-6),
        noise=resid.std(),
        hf_ratio=spec[RADIUS > 0.5].sum() / spec.sum(),
        zero_dx=(dx < 0.5).mean(),
        flat_blocks=(blk < 1.0).mean(),
        n_colours=len(np.unique(rgb.reshape(-1, 3) // 4, axis=0)),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, imgs = train_images()
    df = with_folds(df)
    rows = []
    for i, (p, rgb) in enumerate(zip(df.path, imgs)):
        rows.append({**file_props(p), **pixel_props(np.asarray(rgb))})
        if i % 1000 == 0:
            print(i, flush=True)
    out = pd.concat([df[["image_id", "label", "cluster_id", "fold"]], pd.DataFrame(rows)], axis=1)
    out.to_csv(OUT / "image_stats.csv", index=False)
    print("saved", OUT / "image_stats.csv", out.shape)


if __name__ == "__main__":
    main()
