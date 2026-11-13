"""T1.2 step 2 + T1.3: near-duplicate clusters -> frozen 5-fold StratifiedGroupKFold.

Edge rule (calibrated by eye on review grids, see reports/figures/dups/):
  same source scene if c16 >= 0.93, or (c16 >= 0.90 and pHash distance <= 8),
  or emb >= 0.75 (ConvNeXt-T standardised cosine; catches re-crops and heavy degradations
  that pixel signals miss; over-merges some similar-looking scenes, max cluster 11).
Outputs: data/clusters.csv, data/folds.csv, data/opposite_label_pairs.csv,
         reports/figures/cluster_sizes.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from src.common import CLUSTERS_CSV, FOLDS_CSV, ROOT
from src.data import train_table

N_FOLDS, SEED = 5, 42


def edges(pairs: pd.DataFrame) -> pd.DataFrame:
    keep = (pairs.c16 >= 0.93) | ((pairs.c16 >= 0.90) & (pairs.ham <= 8)) | (pairs.emb >= 0.75)
    return pairs[keep]


def main():
    df = train_table()
    pairs = pd.read_csv(ROOT / "data" / "dup_candidates.csv")
    e = edges(pairs)
    idx = {v: k for k, v in enumerate(df.image_id)}
    a, b = e.a.map(idx).values, e.b.map(idx).values
    n = len(df)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    df["cluster_id"] = comp

    sizes = df.cluster_id.value_counts()
    mixed = df.groupby("cluster_id").label.nunique()
    print(f"edges: {len(e)} | clusters: {len(sizes)} | images in multi-image clusters: "
          f"{int(sizes[sizes > 1].sum())} ({sizes[sizes > 1].sum() / n:.1%})")
    print("cluster size counts:", sizes.value_counts().sort_index().to_dict())
    print(f"clusters with both labels: {(mixed > 1).sum()}")
    df[["image_id", "label", "cluster_id"]].to_csv(CLUSTERS_CSV, index=False)
    e[~e.same_label].to_csv(ROOT / "data" / "opposite_label_pairs.csv", index=False)
    print(f"opposite-label edges: {(~e.same_label).sum()} of {len(e)}")

    fig, ax = plt.subplots(figsize=(5, 3))
    vc = sizes.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_yscale("log"); ax.set_xlabel("cluster size"); ax.set_ylabel("clusters (log)")
    ax.set_title("Near-duplicate cluster sizes (train)")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / "cluster_sizes.png", dpi=120)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(df, df.label, groups=df.cluster_id)):
        df.loc[va, "fold"] = f
    df[["image_id", "label", "cluster_id", "fold"]].to_csv(FOLDS_CSV, index=False)
    print(df.groupby("fold").label.agg(["size", "mean"]))
    print(f"saved {FOLDS_CSV}")


if __name__ == "__main__":
    main()
