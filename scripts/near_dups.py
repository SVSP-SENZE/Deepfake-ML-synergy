"""T1.2 step 1: find candidate near-duplicate pairs in TRAIN only and save review grids.

Signals per pair:
  c16   correlation of z-scored 16x16 grayscale (max over the image and its mirror)
  ham   pHash Hamming distance (64-bit, min over image and mirror)
  c64   correlation of z-scored 64x64 grayscale (computed for candidates only)
Outputs: data/dup_candidates.csv, reports/figures/dups/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common import ROOT, seed_everything
from src.data import train_images

C16_MIN, HAM_MAX = 0.90, 12


def zvec(gray: np.ndarray, s: int) -> np.ndarray:
    v = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def main():
    seed_everything(0)
    df, imgs = train_images()
    n = len(df)
    print(f"{n} train images")
    gray = [cv2.cvtColor(np.ascontiguousarray(imgs[k]), cv2.COLOR_RGB2GRAY) for k in range(n)]
    g16 = np.stack([zvec(g, 16) for g in gray])
    g16f = np.stack([zvec(g[:, ::-1], 16) for g in gray])

    def phash_bits(g):
        return np.array(imagehash.phash(Image.fromarray(g)).hash, dtype=bool).ravel()
    ph = np.stack([phash_bits(g) for g in gray]).astype(np.float32)
    phf = np.stack([phash_bits(np.ascontiguousarray(g[:, ::-1])) for g in gray]).astype(np.float32)

    # compare in row blocks, keep only candidate pairs (upper triangle)
    I, J, C, H = [], [], [], []
    for s0 in range(0, n, 500):
        sl = slice(s0, min(s0 + 500, n))
        c = np.maximum(g16[sl] @ g16.T, g16[sl] @ g16f.T)
        a = ph[sl]
        h = np.minimum(a @ (1 - ph).T + (1 - a) @ ph.T, a @ (1 - phf).T + (1 - a) @ phf.T)
        rows = np.arange(sl.start, sl.stop)[:, None]
        m = ((c >= C16_MIN) | (h <= HAM_MAX)) & (np.arange(n)[None, :] > rows)
        ii, jj = np.nonzero(m)
        I.append(ii + sl.start); J.append(jj); C.append(c[ii, jj]); H.append(h[ii, jj])
    i, j = np.concatenate(I), np.concatenate(J)
    c16v, hamv = np.concatenate(C), np.concatenate(H)
    print(f"{len(i)} candidate pairs (c16>={C16_MIN} or ham<={HAM_MAX})")

    g64 = {k: zvec(gray[k], 64) for k in set(i) | set(j)}
    g64f = {k: zvec(gray[k][:, ::-1], 64) for k in g64}
    c64 = np.array([max(g64[a] @ g64[b], g64[a] @ g64f[b]) for a, b in zip(i, j)])

    lab = df.label.values
    pairs = pd.DataFrame({"a": df.image_id.values[i], "b": df.image_id.values[j],
                          "label_a": lab[i], "label_b": lab[j],
                          "c16": c16v, "ham": hamv.astype(int), "c64": c64})
    pairs["same_label"] = pairs.label_a == pairs.label_b
    out = ROOT / "data" / "dup_candidates.csv"
    pairs.sort_values("c64", ascending=False).to_csv(out, index=False)
    print(f"saved {out}")

    # summary by c64 bin
    bins = [-1, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.01]
    pairs["bin"] = pd.cut(pairs.c64, bins)
    print(pairs.groupby("bin", observed=True).agg(n=("a", "size"), same=("same_label", "mean"),
                                                   ham_med=("ham", "median")))

    # review grids: 10 random pairs per c64 bin
    fig_dir = ROOT / "reports" / "figures" / "dups"
    fig_dir.mkdir(parents=True, exist_ok=True)
    idx = {v: k for k, v in enumerate(df.image_id)}
    rng = np.random.default_rng(0)
    name = {0: "REAL", 1: "FAKE"}
    for b, grp in pairs.groupby("bin", observed=True):
        take = grp.iloc[rng.permutation(len(grp))[:10]]
        tile = 160
        canvas = Image.new("RGB", (tile * 4 + 30, tile * ((len(take) + 1) // 2) + 10), "white")
        d = ImageDraw.Draw(canvas)
        for k, r in enumerate(take.itertuples()):
            x0, y0 = (k % 2) * (tile * 2 + 20), (k // 2) * tile
            for m, (iid, l) in enumerate([(r.a, r.label_a), (r.b, r.label_b)]):
                im = Image.fromarray(np.asarray(imgs[idx[iid]])).resize((tile - 4, tile - 18))
                canvas.paste(im, (x0 + m * tile, y0 + 16))
                d.text((x0 + m * tile + 2, y0 + 2), name[l], fill="red" if l else "green")
            d.text((x0 + 50, y0 + 2), f"c64={r.c64:.3f} h={r.ham}", fill="black")
        lo, hi = b.left, b.right
        canvas.save(fig_dir / f"pairs_c64_{max(lo, 0):.2f}-{min(hi, 1):.2f}.png")
    print(f"grids in {fig_dir}")


if __name__ == "__main__":
    main()
