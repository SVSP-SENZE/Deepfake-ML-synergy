"""T1.2 step 3: embedding-similarity candidates (ConvNeXt-T last-layer features,
each dimension standardised, then cosine).

Adds column `emb` to data/dup_candidates.csv (pairs above EMB_MIN are appended as new rows)
and prints counts by bin so the threshold can be calibrated with scripts/pair_grid.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import CACHE, ROOT
from src.data import train_table

EMB_MIN = 0.55


def main():
    df = train_table()
    f = np.load(CACHE / "feats" / "convnext_t" / "clean_last.npy").astype(np.float32)
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-6
    n = len(f)
    I, J, S = [], [], []
    for s in range(0, n, 1000):
        sim = f[s:s + 1000] @ f.T
        rows = np.arange(s, min(s + 1000, n))[:, None]
        ii, jj = np.nonzero((sim >= EMB_MIN) & (np.arange(n)[None, :] > rows))
        I.append(ii + s); J.append(jj); S.append(sim[ii, jj])
    i, j, s = map(np.concatenate, (I, J, S))
    ids, lab = df.image_id.values, df.label.values
    emb = pd.DataFrame({"a": ids[i], "b": ids[j], "emb": s})

    p = pd.read_csv(ROOT / "data" / "dup_candidates.csv").drop(columns=["emb"], errors="ignore")
    m = p.merge(emb, on=["a", "b"], how="outer")
    new = m.label_a.isna()
    idx = {v: k for k, v in enumerate(ids)}
    m.loc[new, "label_a"] = m.loc[new, "a"].map(lambda x: lab[idx[x]])
    m.loc[new, "label_b"] = m.loc[new, "b"].map(lambda x: lab[idx[x]])
    m["same_label"] = m.label_a == m.label_b
    m[["c16", "c64"]] = m[["c16", "c64"]].fillna(0)
    m["ham"] = m.ham.fillna(64).astype(int)
    m["emb"] = m.emb.fillna(0)
    m = m.astype({"label_a": int, "label_b": int})

    # whole-population cosine for every pair, for reference
    m.to_csv(ROOT / "data" / "dup_candidates.csv", index=False)
    pixel_edge = (m.c16 >= 0.93) | ((m.c16 >= 0.90) & (m.ham <= 8))
    m["bin"] = pd.cut(m.emb, [-1, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.01])
    print(m.groupby("bin", observed=True).agg(n=("a", "size"), pixel_edge=("c16", lambda x: pixel_edge[x.index].mean()),
                                              same_label=("same_label", "mean")))


if __name__ == "__main__":
    main()
