"""T2.3 / T2.4: fine-tune a backbone on grouped folds, with the robustness proxy.

Usage: python scripts/finetune_cv.py --backbone convnext_t --folds 0 --epochs 10 --lr 1e-4
       [--regime full|layernorm|head] [--shift 8] [--name e1_convnext]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.common import ROOT, log_result, seed_everything, start_run
from src.data import train_images, with_folds
from src.metrics import cluster_bootstrap_auc
from src.robustness import summarise
from src.train import train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_t")
    ap.add_argument("--regime", default="full")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--shift", type=int, default=8)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None)
    ap.add_argument("--hypothesis", default="T2.3 fine-tune baseline")
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument("--head_lr_mult", type=float, default=10)
    ap.add_argument("--kill_below", type=float, nargs=2, default=None,
                    help="EPOCH AUC: stop a fold if val AUC < AUC at EPOCH")
    ap.add_argument("--no_robustness", action="store_true")
    ap.add_argument("--degrade", default="off", choices=["off", "weak", "medium", "strong", "xstrong"])
    ap.add_argument("--loss", default="bce", choices=["bce", "gce", "sce"])
    ap.add_argument("--gce_q", type=float, default=0.7)
    ap.add_argument("--sce_beta", type=float, default=1.0)
    ap.add_argument("--prune", action="store_true",
                    help="drop cleanlab-flagged images (data/label_issues.csv) from TRAINING folds only")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off")
    ap.add_argument("--tta_check", action="store_true", help="also log AUCs with flip TTA")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items()
           if k not in ("name", "hypothesis", "save_weights", "no_robustness")}
    cfg["robustness"] = not args.no_robustness
    if cfg["kill_below"]:
        cfg["kill_below"] = [int(cfg["kill_below"][0]), cfg["kill_below"][1]]
    cfg["hflip"] = True
    seed_everything(args.seed)
    run = start_run(args.name or f"ft_{args.backbone}_{args.regime}", cfg, args.seed)
    logf = open(run / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    df, imgs = train_images()
    df = with_folds(df)
    y, folds = df.label.values, df.fold.values
    flagged = np.zeros(len(y), bool)
    if args.prune:
        li = pd.read_csv(ROOT / "data" / "label_issues.csv").set_index("image_id")
        flagged = df.image_id.map(li.flagged).fillna(False).values.astype(bool)
        log(f"pruning {flagged.sum()} flagged images from training folds")
    t0 = time.time()
    oof = np.full(len(y), np.nan)
    var = {}
    curves = {}
    for f in args.folds:
        log(f"fold {f}")
        tr, va = np.flatnonzero((folds != f) & ~flagged), np.flatnonzero(folds == f)
        r = train_fold(cfg, imgs, y, tr, va, seed=args.seed + f, log=log)
        oof[va] = r["val_logit"]
        for v, l in r.get("variant_logit", {}).items():
            var.setdefault(v, np.full(len(y), np.nan))[va] = l
        curves[f] = r["curve"]
        if args.save_weights:
            torch.save(r["model"].state_dict(), run / f"fold{f}.pt")
        del r
        torch.cuda.empty_cache()

    m = ~np.isnan(oof)
    auc = roc_auc_score(y[m], oof[m])
    fold_aucs = [roc_auc_score(y[folds == f], oof[folds == f]) for f in args.folds]
    lo, hi = cluster_bootstrap_auc(y[m], oof[m], df.cluster_id.values[m], n_boot=300)
    rob = summarise(auc, {v: roc_auc_score(y[m], l[m]) for v, l in var.items()}) if var else {}
    np.save(run / "oof_logit.npy", oof)
    if var:
        np.savez(run / "oof_variant_logits.npz", **var)
    (run / "curves.json").write_text(json.dumps(curves, indent=1))
    log(f"AUC {auc:.4f} [{lo:.3f}-{hi:.3f}] folds {np.round(fold_aucs, 4).tolist()}")
    for k, v in rob.items():
        log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    log_result(run, args.hypothesis, cfg, args.seed, cv_auc=auc,
               cv_auc_std=float(np.std(fold_aucs)) if len(fold_aucs) > 1 else "",
               robust_mean_auc=rob.get("robust_mean_auc", ""), robust_worst_auc=rob.get("robust_worst_auc", ""),
               runtime_s=round(time.time() - t0, 1),
               notes=f"folds={args.folds}; CI {lo:.3f}-{hi:.3f}; worst={rob.get('worst_variant')}")


if __name__ == "__main__":
    main()
