"""T3.3 step 1: confident learning on 5-fold OOF predictions (ViT-B + ConvNeXt-T, Day 3 runs).

Outputs: data/label_issues.csv (image_id, label, p_fake, flagged, rank),
         reports/figures/label_issues_*.png review grids.
Caveat: each image's OOF prediction comes from models trained on the other 4 folds, so when
the flags are used to prune training data for fold f, fold f's labels have mildly influenced
which training images are dropped. Noted in the report.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from cleanlab.count import estimate_joint
from cleanlab.filter import find_label_issues
from PIL import Image, ImageDraw

from src.common import ROOT
from src.data import train_images, with_folds


def sig(x):
    return 1 / (1 + np.exp(-x))


def grid(imgs, rows, path, title_fn, n=16):
    tile = 150
    canvas = Image.new("RGB", (tile * 8, tile * ((min(n, len(rows)) + 7) // 8) + 4), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(rows[:n].itertuples()):
        x, y = (k % 8) * tile, (k // 8) * tile
        canvas.paste(Image.fromarray(np.asarray(imgs[r.idx])).resize((tile - 4, tile - 18)), (x, y + 16))
        d.text((x + 2, y + 2), title_fn(r), fill="red" if r.label else "green")
    canvas.save(path)


def main():
    df, imgs = train_images()
    df = with_folds(df)
    y = df.label.values
    runs = [glob.glob(str(ROOT / "experiments" / "runs" / f"*{n}"))[0]
            for n in ("d3_vit_full_5fold", "d3_cnx_full_5fold")]
    p = np.mean([sig(np.load(Path(r) / "oof_logit.npy")) for r in runs], axis=0)
    probs = np.stack([1 - p, p], axis=1)

    joint = estimate_joint(y, probs)
    noise_rate = 1 - np.trace(joint)
    print("estimated joint P(given label, true label):\n", joint.round(4))
    print(f"estimated label-noise rate: {noise_rate:.3f}  "
          f"(real->fake {joint[0, 1]:.3f}, fake->real {joint[1, 0]:.3f})")

    flagged = find_label_issues(y, probs, filter_by="prune_by_noise_rate", n_jobs=1)
    order = find_label_issues(y, probs, return_indices_ranked_by="self_confidence", n_jobs=1)
    rank = np.full(len(y), -1)
    rank[order] = np.arange(len(order))
    print(f"flagged: {flagged.sum()} ({flagged.mean():.1%}); "
          f"given REAL {flagged[y == 0].sum()}, given FAKE {flagged[y == 1].sum()}")

    out = df[["image_id", "label", "cluster_id", "fold"]].copy()
    out["p_fake"], out["flagged"], out["rank"] = p, flagged, rank
    out.to_csv(ROOT / "data" / "label_issues.csv", index=False)

    out["idx"] = np.arange(len(out))
    name = {0: "REAL", 1: "FAKE"}
    top = out[out.flagged].sort_values("rank")
    fig = ROOT / "reports" / "figures"
    grid(imgs, top[top.label == 0], fig / "label_issues_given_real.png",
         lambda r: f"{name[r.label]} p_fake={r.p_fake:.2f}")
    grid(imgs, top[top.label == 1], fig / "label_issues_given_fake.png",
         lambda r: f"{name[r.label]} p_fake={r.p_fake:.2f}")
    # how flagged rate relates to mixed-label clusters
    mixed = out.groupby("cluster_id").label.transform("nunique") > 1
    print(f"flag rate: in mixed-label clusters {out.flagged[mixed].mean():.1%} "
          f"({mixed.sum()} imgs) vs elsewhere {out.flagged[~mixed].mean():.1%}")


if __name__ == "__main__":
    main()
