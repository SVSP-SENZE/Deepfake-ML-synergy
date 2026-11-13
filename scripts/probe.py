"""T2.2 frozen-feature probe: logistic regression on cached backbone features,
grouped 5-fold CV, plus the robustness proxy (same fold models scored on perturbed features).

Usage: python scripts/probe.py convnext_t vit_b16 ... [--layers last mid both]
Run scripts/extract_features.py first.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import CACHE, log_result, seed_everything, start_run
from src.data import train_images, with_folds
from src.metrics import cluster_bootstrap_auc
from src.robustness import PERTURBATIONS, summarise

CS = [0.0001, 0.0003, 0.001, 0.01]


def load(key, variant, layers):
    d = CACHE / "feats" / key
    parts = {"last": ["last"], "mid": ["mid"], "both": ["last", "mid"]}[layers]
    return np.concatenate([np.load(d / f"{variant}_{p}.npy").astype(np.float32) for p in parts], axis=1)


def run_probe(key, layers, df, C):
    y, folds = df.label.values, df.fold.values
    X = load(key, "clean", layers)
    oof = np.zeros(len(y))
    models = []
    for f in range(5):
        tr, va = folds != f, folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        models.append(clf)
    return oof, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbones", nargs="+")
    ap.add_argument("--layers", nargs="+", default=["last", "mid"])
    args = ap.parse_args()
    seed_everything(0)
    df, _ = train_images()
    df = with_folds(df)
    y, folds = df.label.values, df.fold.values
    rows = []
    for key in args.backbones:
        for layers in args.layers:
            t0 = time.time()
            # pick C on OOF AUC (screening only; noted as mild optimism)
            res = {C: run_probe(key, layers, df, C) for C in CS}
            C = max(res, key=lambda c: roc_auc_score(y, res[c][0]))
            oof, models = res[C]
            auc = roc_auc_score(y, oof)
            lo, hi = cluster_bootstrap_auc(y, oof, df.cluster_id, n_boot=300)
            var_auc = {}
            for v in PERTURBATIONS:
                Xv = load(key, v, layers)
                pv = np.zeros(len(y))
                for f, clf in enumerate(models):
                    pv[folds == f] = clf.predict_proba(Xv[folds == f])[:, 1]
                var_auc[v] = roc_auc_score(y, pv)
            rob = summarise(auc, var_auc)
            cfg = {"probe": "logreg", "backbone": key, "layers": layers, "C": C}
            run = start_run(f"probe_{key}_{layers}", cfg, 0)
            np.save(run / "oof.npy", oof)
            log_result(run, "T2.2 frozen-feature probe", cfg, 0, cv_auc=auc,
                       robust_mean_auc=rob["robust_mean_auc"], robust_worst_auc=rob["robust_worst_auc"],
                       runtime_s=round(time.time() - t0, 1),
                       notes=f"CI {lo:.3f}-{hi:.3f}; worst={rob['worst_variant']}")
            rows.append((key, layers, C, auc, lo, hi, rob["robust_mean_auc"], rob["robust_worst_auc"],
                         rob["worst_variant"]))
            print(f"{key:11s} {layers:5s} C={C:<6} AUC {auc:.4f} [{lo:.3f}-{hi:.3f}] "
                  f"robust mean {rob['robust_mean_auc']:.4f} worst {rob['robust_worst_auc']:.4f} "
                  f"({rob['worst_variant']})", flush=True)
    print("\n| backbone | layers | C | AUC | 95% CI | robust mean | robust worst |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.3f}-{r[5]:.3f} | {r[6]:.4f} | {r[7]:.4f} ({r[8]}) |")


if __name__ == "__main__":
    main()
