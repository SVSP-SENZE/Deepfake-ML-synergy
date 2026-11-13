"""T4.2: combine members on out-of-fold logits, clean + robustness proxy.

Usage: python scripts/ensemble_eval.py vit=d5_vit_prune,d6_vit_prune_f234 cnx=d5_cnx_prune,d6_cnx_prune_f234 ...
Each member = one or more run names whose fold subsets are merged. Evaluated on images every member covers.
Standardisation (mean/std of each member's clean OOF logits) is fitted once on OOF and reused unchanged for
every perturbed variant: nothing depends on the test set. Stats are written to experiments/ensemble_stats.json.
"""
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.common import ROOT
from src.data import train_table, with_folds
from src.metrics import cluster_bootstrap_auc
from src.robustness import PERTURBATIONS, summarise


def load_member(names):
    oof, var = None, {}
    for n in names:
        d = Path(sorted(glob.glob(str(ROOT / "experiments" / "runs" / f"*_{n}")))[-1])
        o = np.load(d / "oof_logit.npy")
        oof = o if oof is None else np.where(np.isnan(oof), o, oof)
        z = np.load(d / "oof_variant_logits.npz")
        for v in z.files:
            var[v] = z[v] if v not in var else np.where(np.isnan(var[v]), z[v], var[v])
    return oof, var


def combine(logits, stats, how, w=None):
    w = np.ones(len(logits)) / len(logits) if w is None else np.asarray(w)
    if how == "zlogit":
        return sum(wi * (l - m) / s for wi, l, (m, s) in zip(w, logits, stats))
    if how == "prob":
        return sum(wi / (1 + np.exp(-l)) for wi, l in zip(w, logits))
    if how == "rank":
        return sum(wi * rankdata(l) for wi, l in zip(w, logits))
    raise ValueError(how)


def evaluate(y, mask, clean, var, stats, how, w=None):
    c = combine([l[mask] for l in clean], stats, how, w)
    auc = roc_auc_score(y[mask], c)
    vauc = {v: roc_auc_score(y[mask], combine([vl[v][mask] for vl in var], stats, how, w)) for v in PERTURBATIONS}
    return c, summarise(auc, vauc)


def main():
    members = dict(a.split("=") for a in sys.argv[1:])
    df = with_folds(train_table())
    y, groups = df.label.values, df.cluster_id.values
    clean, var = {}, {}
    for k, names in members.items():
        clean[k], var[k] = load_member(names.split(","))
    mask = np.all([~np.isnan(clean[k]) for k in members], axis=0)
    print(f"evaluated on {mask.sum()} images, folds {sorted(set(df.fold[mask]))}")
    stats = {k: (float(clean[k][mask].mean()), float(clean[k][mask].std())) for k in members}

    rows = []
    def row(name, keys, how, w=None):
        c, r = evaluate(y, mask, [clean[k] for k in keys], [var[k] for k in keys], [stats[k] for k in keys], how, w)
        lo, hi = cluster_bootstrap_auc(y[mask], c, groups[mask], n_boot=200)
        rows.append((name, r["clean_auc"] if "clean_auc" in r else roc_auc_score(y[mask], c), lo, hi,
                     r["seen_mean_auc"], r["unseen_mean_auc"], r["robust_mean_auc"], r["robust_worst_auc"]))

    for k in members:
        row(k, [k], "zlogit")
    keys = list(members)
    for n in range(2, len(keys) + 1):
        for sub in itertools.combinations(keys, n):
            for how in ("zlogit", "prob", "rank"):
                row("+".join(sub) + f" [{how}]", sub, how)
    # correlation of members' clean OOF logits
    print("Spearman between members:")
    for a, b in itertools.combinations(keys, 2):
        from scipy.stats import spearmanr
        print(f"  {a}-{b}: {spearmanr(clean[a][mask], clean[b][mask])[0]:.3f}")
    print(f"{'ensemble':40s} clean  [CI]          seen   unseen robust worst")
    for r in rows:
        print(f"{r[0]:40s} {r[1]:.4f} [{r[2]:.3f}-{r[3]:.3f}] {r[4]:.4f} {r[5]:.4f} {r[6]:.4f} {r[7]:.4f}")
    (ROOT / "experiments" / "ensemble_stats.json").write_text(json.dumps(
        {"members": members, "zlogit_stats": stats, "n_images": int(mask.sum())}, indent=1))


if __name__ == "__main__":
    main()
