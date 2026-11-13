"""D1 exit check: trivial model through grouped 5-fold CV + robustness proxy,
and the shuffled-label negative control (T1.7).

Model: logistic regression on 32x32 RGB pixels plus simple image statistics.
Usage: python scripts/baseline_cv.py [--shuffle-labels] [--seed 42]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import log_result, seed_everything, start_run
from src.data import train_images, with_folds
from src.metrics import accuracy_points, basic, cluster_bootstrap_auc, robustness_points
from src.robustness import PERTURBATIONS, perturb_all, summarise


def features(imgs) -> np.ndarray:
    out = []
    for im in imgs:
        im = np.asarray(im)
        small = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        stats = [cv2.Laplacian(g, cv2.CV_64F).var() ** 0.5, g.mean(), g.std(),
                 hsv[..., 1].mean(), hsv[..., 1].std(), *im.reshape(-1, 3).mean(0)]
        out.append(np.concatenate([small, np.asarray(stats, np.float32)]))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle-labels", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    cfg = {"model": "logreg_pixels32_stats", "C": 0.01, "shuffle_labels": args.shuffle_labels}
    run = start_run("baseline_shuffled" if args.shuffle_labels else "baseline", cfg, args.seed)
    t0 = time.time()

    df, imgs = train_images()
    df = with_folds(df)
    y = df.label.values
    X = features(imgs)
    y_train = np.random.default_rng(args.seed).permutation(y) if args.shuffle_labels else y

    oof = np.zeros(len(df))
    oof_pert = {k: np.zeros(len(df)) for k in PERTURBATIONS}
    fold_aucs = []
    for f in range(5):
        tr, va = np.flatnonzero(df.fold != f), np.flatnonzero(df.fold == f)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=2000))
        clf.fit(X[tr], y_train[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        fold_aucs.append(basic(y[va], oof[va])["auc"])
        for name in PERTURBATIONS:
            oof_pert[name][va] = clf.predict_proba(features(perturb_all(imgs, va, name)))[:, 1]
        print(f"fold {f}: auc {fold_aucs[-1]:.4f}")

    m = basic(y, oof)
    lo, hi = cluster_bootstrap_auc(y, oof, df.cluster_id)
    rob = summarise(m["auc"], {k: basic(y, v)["auc"] for k, v in oof_pert.items()})
    np.save(run / "oof.npy", oof)
    print(f"OOF AUC {m['auc']:.4f} (95% cluster-bootstrap CI {lo:.4f}-{hi:.4f}), "
          f"fold mean {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    for k, v in rob.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"rubric proxy: accuracy {accuracy_points(m['auc']):.1f}/50, "
          f"robustness {robustness_points([rob[f'auc_{k}'] for k in PERTURBATIONS]):.1f}/20")

    hyp = ("T1.7 negative control: shuffled train labels should give ~0.5 AUC"
           if args.shuffle_labels else "D1 trivial baseline: LR on 32px pixels + stats")
    log_result(run, hyp, cfg, args.seed, cv_auc=m["auc"], cv_auc_std=float(np.std(fold_aucs)),
               robust_mean_auc=rob["robust_mean_auc"], robust_worst_auc=rob["robust_worst_auc"],
               runtime_s=round(time.time() - t0, 1),
               notes=f"CI {lo:.3f}-{hi:.3f}; worst={rob['worst_variant']}")
    if args.shuffle_labels:
        assert abs(m["auc"] - 0.5) < 0.04, f"negative control FAILED: AUC {m['auc']:.4f}"
        print("negative control PASSED")


if __name__ == "__main__":
    main()
