"""EDA step 1b: pixelation and occlusion-patch detectors (added after the first features failed the eye check)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.common import ROOT
from src.data import train_images


def blockiness(gray):
    """Pixelation leaves gradients only on a regular grid: best ratio of |grad| on grid lines vs elsewhere."""
    best = 1.0
    for ax in (0, 1):
        g = np.abs(np.diff(gray, axis=ax)).mean(axis=1 - ax)
        for k in range(3, 17):
            on = np.zeros_like(g, dtype=bool)
            for off in range(k):
                m = np.zeros_like(on); m[off::k] = True
                r = g[m].mean() / (g[~m].mean() + 1e-3)
                best = max(best, r) if ax == 0 else best
                if ax == 1:
                    best = max(best, r)
    return best


def patch_score(rgb):
    """Largest near-constant region that is a filled rectangle (synthetic occlusion)."""
    f = rgb.astype(np.float32)
    var = sum(cv2.blur(c * c, (5, 5)) - cv2.blur(c, (5, 5)) ** 2 for c in cv2.split(f))
    mask = (var < 1.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    best_area, best_fill = 0.0, 0.0
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if a < 120 or w < 8 or h < 8:
            continue
        fill = a / (w * h)
        touches_edge = x == 0 or y == 0 or x + w == 224 or y + h == 224
        if fill > 0.85 and not (touches_edge and a > 0.2 * 224 * 224):
            if a > best_area:
                best_area, best_fill = a / (224 * 224), fill
    return best_area, best_fill


def patch_edge(rgb):
    """Contrast between a small uniform rectangle and the ring around it. >= 15 = synthetic occlusion patch
    (checked by eye: almost every sampled image above 15 has a visible patch; fainter patches below are missed)."""
    f = rgb.astype(np.float32)
    var = sum(cv2.blur(c * c, (3, 3)) - cv2.blur(c, (3, 3)) ** 2 for c in cv2.split(f))
    n, _, st, _ = cv2.connectedComponentsWithStats((var < 2.0).astype(np.uint8), connectivity=4)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = 0.0
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if a < 60 or w < 7 or h < 7 or a / (w * h) < 0.8 or max(w, h) / min(w, h) > 3 or w * h > 0.15 * 224 * 224:
            continue
        inner = g[y + 1:y + h - 1, x + 1:x + w - 1].mean()
        y0, y1, x0, x1 = max(y - 3, 0), min(y + h + 3, 224), max(x - 3, 0), min(x + w + 3, 224)
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        if ring.any():
            best = max(best, abs(g[y0:y1, x0:x1][ring].mean() - inner))
    return best


def main():
    d = pd.read_csv(ROOT / "experiments" / "eda" / "image_stats.csv")
    _, imgs = train_images()
    rows = []
    for i in range(len(d)):
        rgb = np.asarray(imgs[i])
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        a, f = patch_score(rgb)
        rows.append(dict(blockiness=blockiness(g), patch_area=a, patch_fill=f, patch_edge=patch_edge(rgb)))
    for c in ("blockiness", "patch_area", "patch_fill", "patch_edge"):
        d[c] = [r[c] for r in rows]
    d.to_csv(ROOT / "experiments" / "eda" / "image_stats.csv", index=False)
    print(d[["blockiness", "patch_area"]].quantile([.5, .8, .9, .95, .98]).round(3))
    print("patch>0", (d.patch_area > 0).mean())


if __name__ == "__main__":
    main()
