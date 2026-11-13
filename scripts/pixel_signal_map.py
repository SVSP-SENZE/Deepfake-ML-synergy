"""Diagnostic: where does the 32px-pixel logistic-regression signal come from?
Saves the class-mean difference image and the LR weight map, and grouped-CV AUC
for pixel-only, stats-only and centre/border masks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, seed_everything
from src.data import train_images, with_folds


def cv_auc(X, y, folds, C=0.01):
    oof = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof)


def main():
    seed_everything(0)
    df, imgs = train_images()
    df = with_folds(df)
    y, folds = df.label.values, df.fold.values
    px = np.stack([cv2.resize(np.asarray(im), (32, 32), interpolation=cv2.INTER_AREA)
                   for im in imgs]).astype(np.float32) / 255           # N,32,32,3
    gray = px.mean(-1)
    yy, xx = np.mgrid[:32, :32]
    centre = (np.abs(yy - 15.5) < 8) & (np.abs(xx - 15.5) < 8)

    res = {
        "rgb32 all": lambda: px.reshape(len(y), -1),
        "gray32 all": lambda: gray.reshape(len(y), -1),
        "rgb32 centre 16x16 only": lambda: px[:, centre].reshape(len(y), -1),
        "rgb32 border only": lambda: px[:, ~centre].reshape(len(y), -1),
        "rgb per-image normalised": lambda: ((px - px.mean((1, 2, 3), keepdims=True))
                                             / (px.std((1, 2, 3), keepdims=True) + 1e-6)).reshape(len(y), -1),
        "rgb8 all": lambda: np.stack([cv2.resize(p, (8, 8), interpolation=cv2.INTER_AREA)
                                      for p in px]).reshape(len(y), -1),
    }
    for k, make in res.items():
        print(f"{k:28s} grouped-CV AUC {cv_auc(make(), y, folds):.4f}", flush=True)

    diff = px[y == 1].mean(0) - px[y == 0].mean(0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(gray.reshape(len(y), -1), y)
    w = clf[-1].coef_.reshape(32, 32)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    ax[0].imshow(px[y == 0].mean(0)); ax[0].set_title("mean REAL")
    ax[1].imshow(px[y == 1].mean(0)); ax[1].set_title("mean FAKE")
    m = np.abs(diff.mean(-1)).max()
    im2 = ax[2].imshow(diff.mean(-1), cmap="bwr", vmin=-m, vmax=m); ax[2].set_title("FAKE - REAL (gray)")
    plt.colorbar(im2, ax=ax[2])
    m = np.abs(w).max()
    im3 = ax[3].imshow(w, cmap="bwr", vmin=-m, vmax=m); ax[3].set_title("LR weights (gray32)")
    plt.colorbar(im3, ax=ax[3])
    for a in ax: a.axis("off")
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "pixel_signal_map.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print("per-channel mean FAKE-REAL:", diff.mean((0, 1)).round(4))


if __name__ == "__main__":
    main()
