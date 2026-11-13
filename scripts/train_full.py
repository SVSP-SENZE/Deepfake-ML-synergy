"""T4.4 final training: each ensemble member on ALL training images (cleanlab-flagged pruned), N seeds.

Usage: python scripts/train_full.py [--seeds 0 1 2] [--members vit cnx] [--out weights]
Writes <out>/<member>_s<seed>.pt (fp16 state dict) and <out>/ensemble.json (members, recipe,
ensemble weights, z-logit stats from 5-fold OOF, TTA flag). Recipes are frozen here from Days 3-6.
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

from src.common import ROOT, seed_everything
from src.data import train_images
from src.train import train_fold

RECIPES = {  # frozen: see reports/D3-D6_notes.md
    "vit": {"backbone": "vit_b16", "regime": "full", "lr": 5e-5, "degrade": "strong", "weight": 0.7},
    "cnx": {"backbone": "convnext_t", "regime": "full", "lr": 1e-4, "degrade": "xstrong", "weight": 0.3},
}
COMMON = {"epochs": 4, "wd": 0.05, "batch_size": 64, "shift": 8, "head_lr_mult": 10, "hflip": True,
          "loss": "bce", "label_smoothing": 0.0, "ema": 0.0, "robustness": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--members", nargs="+", default=list(RECIPES))
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)

    df, imgs = train_images()
    y = df.label.values
    li = pd.read_csv(ROOT / "data" / "label_issues.csv").set_index("image_id")
    keep = ~df.image_id.map(li.flagged).fillna(False).values.astype(bool)
    tr = np.flatnonzero(keep)
    print(f"training on {len(tr)} of {len(y)} images ({(~keep).sum()} pruned)", flush=True)

    stats = json.loads((ROOT / "experiments" / "ensemble_stats.json").read_text())["zlogit_stats"]
    log = open(out / "train_full.log", "a")

    def p(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    for m in args.members:
        for s in args.seeds:
            cfg = {**COMMON, **{k: v for k, v in RECIPES[m].items() if k != "weight"}}
            seed = 1000 + s  # distinct from the CV seeds
            seed_everything(seed)
            p(f"== {m} seed {seed}")
            t0 = time.time()
            r = train_fold(cfg, imgs, y, tr, np.array([], dtype=int), seed=seed, log=p)
            sd = {k: v.half() if v.is_floating_point() else v for k, v in r["model"].state_dict().items()}
            torch.save(sd, out / f"{m}_s{seed}.pt")
            p(f"   saved {m}_s{seed}.pt ({time.time() - t0:.0f}s)")
            del r
            torch.cuda.empty_cache()

    files = sorted(f.name for f in out.glob("*.pt"))
    cfg_out = {
        "image_size": 224, "resize": "PIL bilinear to 224x224, RGB", "tta_flip": True,
        "combine": "sum_m weight_m * (mean_over_seeds(logit_m) - mean_m) / std_m; confidence = sigmoid",
        "members": {m: {**RECIPES[m], "files": [f for f in files if f.startswith(m + "_")],
                        "zlogit_mean": stats[m][0], "zlogit_std": stats[m][1]} for m in RECIPES},
        "train": {"n_images": int(len(tr)), "pruned": int((~keep).sum()), **COMMON},
    }
    (out / "ensemble.json").write_text(json.dumps(cfg_out, indent=1))
    p(f"wrote ensemble.json; files: {files}")


if __name__ == "__main__":
    main()
