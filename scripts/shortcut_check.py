"""Does a model just re-learn the coarse layout cue found on Day 1?

Reference = logistic regression on 8x8 RGB thumbnails (grouped OOF, current folds).
For each run's OOF predictions report:
  - AUC of the run and of the reference
  - Spearman rank correlation between them
  - AUC of the run *within* quintiles of the reference score (signal beyond the cue)
Usage: python scripts/shortcut_check.py experiments/runs/<run1> [<run2> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import CACHE
from src.data import train_images, with_folds


def reference_oof(imgs, y, folds):
    p = CACHE / f"ref_thumb8_oof_{hash(tuple(folds[:50]))}.npy"
    if p.exists():
        return np.load(p)
    X = np.stack([cv2.resize(np.asarray(im), (8, 8), interpolation=cv2.INTER_AREA)
                  for im in imgs]).reshape(len(y), -1).astype(np.float32)
    oof = np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[tr], y[tr])
        oof[va] = clf.decision_function(X[va])
    np.save(p, oof)
    return oof


def load_oof(run: Path):
    for name in ("oof_logit.npy", "oof.npy"):
        if (run / name).exists():
            return np.load(run / name)
    raise FileNotFoundError(run)


def main():
    df, imgs = train_images()
    df = with_folds(df)
    y, folds = df.label.values, df.fold.values
    ref = reference_oof(imgs, y, folds)
    q = pd.qcut(ref, 5, labels=False)
    print(f"reference 8x8 thumbnail AUC {roc_auc_score(y, ref):.4f}")
    print("| run | AUC | Spearman vs ref | mean AUC within ref quintiles |")
    print("|---|---|---|---|")
    for r in sys.argv[1:]:
        p = load_oof(Path(r))
        m = ~np.isnan(p)
        within = [roc_auc_score(y[m & (q == k)], p[m & (q == k)]) for k in range(5)
                  if len(set(y[m & (q == k)])) == 2]
        print(f"| {Path(r).name} | {roc_auc_score(y[m], p[m]):.4f} | {spearmanr(p[m], ref[m])[0]:.3f} | "
              f"{np.mean(within):.4f} |")


if __name__ == "__main__":
    main()
