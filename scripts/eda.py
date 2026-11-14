"""EDA step 2: figures and tables for reports/EDA.md.

Inputs: experiments/eda/image_stats.csv (scripts/eda_features.py), the decoded training cache, data/*.csv,
cached ConvNeXt-T features and the out-of-fold ensemble logits. Training images only; the test set is never read.
Outputs: reports/figures/eda/*.png and experiments/eda/eda_numbers.json (every number quoted in EDA.md).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT
from src.data import train_images

FIG = ROOT / "reports" / "figures" / "eda"
EDA = ROOT / "experiments" / "eda"
C_REAL, C_FAKE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
N = {}  # numbers for the report

FEATURES = {  # column -> readable name
    "brightness": "brightness (mean luma)", "contrast": "contrast (luma std)", "saturation": "saturation (HSV S)",
    "colourfulness": "colourfulness (Hasler)", "chroma_spread": "channel spread (0 = grayscale)",
    "entropy": "luma histogram entropy", "clip_dark": "fraction near-black", "clip_bright": "fraction near-white",
    "sharpness": "sharpness (log Laplacian var)", "noise": "noise (median-filter residual)",
    "hf_ratio": "high-frequency energy ratio", "zero_dx": "flat horizontal steps (low detail)",
    "flat_blocks": "flat 8x8 blocks (smooth area)", "blockiness": "grid-aligned gradients",
    "patch_edge": "occlusion-patch edge contrast", "n_colours": "distinct colours", "file_kb": "file size (KB)",
}


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130, facecolor=BG)
    plt.close(fig)
    print("saved", name)


def grouped_lr_auc(X, y, folds):
    oof = np.zeros(len(y))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof)


# ---------------------------------------------------------------- degradation tags
def tag(d):
    """Heuristic degradation tags. Thresholds set by eye on the grids in figures/eda/tag_*.png."""
    t = pd.DataFrame(index=d.index)  # pixelation is not tagged: no detector passed the eye check
    t["grayscale"] = d.chroma_spread < 2.5
    t["noisy"] = d.noise > 9
    t["blurred"] = d.sharpness < 1.2
    t["patched"] = d.patch_edge >= 15
    t["tone_extreme"] = (d.clip_dark + d.clip_bright > 0.25) | (d.contrast < 25)
    t["none_detected"] = ~t.any(axis=1)
    return t


def ensemble_oof(df):
    st = json.loads((ROOT / "experiments" / "ensemble_stats.json").read_text())
    z = 0
    for k, w in (("vit", 0.7), ("cnx", 0.3)):
        oof = None
        for n in st["members"][k].split(","):
            d = Path(sorted(glob.glob(str(ROOT / "experiments" / "runs" / f"*_{n}")))[-1])
            o = np.load(d / "oof_logit.npy")
            oof = o if oof is None else np.where(np.isnan(oof), o, oof)
        m, s = st["zlogit_stats"][k]
        z = z + w * (oof - m) / s
    return 1 / (1 + np.exp(-z))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(EDA / "image_stats.csv")
    dfi, imgs = train_images()
    assert (dfi.image_id.values == d.image_id.values).all()
    y, folds = d.label.values, d.fold.values
    meta = pd.read_csv(ROOT.parent / "hackathon_dataset" / "train_metadata.csv")

    # ---------------- 1. integrity and balance
    N["n_images"] = len(d)
    N["n_real"], N["n_fake"] = int((y == 0).sum()), int((y == 1).sum())
    N["unique_ids"] = int(meta.image_id.nunique())
    N["files_on_disk"] = sum(len(glob.glob(str(ROOT.parent / "hackathon_dataset" / "train" / c / "*")))
                             for c in ("real", "fake"))
    N["file_props"] = {c: d[c].astype(str).value_counts().to_dict()
                       for c in ("width", "height", "mode", "fmt", "q_dc", "qtable", "has_exif", "progressive")}
    N["fold_table"] = d.groupby("fold").agg(n=("label", "size"), fake_rate=("label", "mean")).round(3).reset_index().to_dict("records")

    # ---------------- 2. univariate feature separability
    rows = []
    for c, name in FEATURES.items():
        a = roc_auc_score(y, d[c])
        rows.append(dict(feature=c, name=name, auc=a, sep=max(a, 1 - a), higher_in="fake" if a > 0.5 else "real",
                         real_median=d[c][y == 0].median(), fake_median=d[c][y == 1].median()))
    uni = pd.DataFrame(rows).sort_values("sep", ascending=False)
    N["univariate"] = uni.round(4).to_dict("records")
    X = d[list(FEATURES)].values.astype(float)
    N["lr_all_stats_auc"] = grouped_lr_auc(X, y, folds)

    fig, ax = plt.subplots(3, 6, figsize=(17, 8.2))
    for a, r in zip(ax.ravel(), uni.itertuples()):
        v = d[r.feature]
        lo, hi = np.percentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi, 40) if hi > lo else 20
        a.hist(v[y == 0], bins=bins, color=C_REAL, alpha=0.55, density=True, label="real")
        a.hist(v[y == 1], bins=bins, color=C_FAKE, alpha=0.55, density=True, label="fake")
        style(a, f"{r.name}\nAUC {r.auc:.3f}")
        a.set_yticks([])
    ax[0, 0].legend(fontsize=7.5, frameon=False)
    for a in ax.ravel()[len(uni):]:
        a.axis("off")
    fig.suptitle("Per-image statistics by class (sorted by separability; AUC of the raw value)", fontsize=11, x=0.01, ha="left")
    save(fig, "feature_distributions.png")

    # ---------------- 3. degradation tags
    t = tag(d)
    tags = list(t.columns)
    N["tags"] = {k: dict(n=int(t[k].sum()), share=round(float(t[k].mean()), 4),
                         share_real=round(float(t[k][y == 0].mean()), 4), share_fake=round(float(t[k][y == 1].mean()), 4),
                         fake_rate=round(float(y[t[k]].mean()), 4) if t[k].any() else None) for k in tags}
    N["n_tags_hist"] = t.drop(columns="none_detected").sum(axis=1).value_counts().sort_index().to_dict()
    rng = np.random.default_rng(0)
    for k in tags:
        idx = np.flatnonzero(t[k].values)
        pick = rng.choice(idx, min(12, len(idx)), replace=False) if len(idx) else []
        fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
        for a in ax.ravel():
            a.axis("off")
        for a, i in zip(ax.ravel(), pick):
            a.imshow(imgs[i]); a.set_title("FAKE" if y[i] else "REAL", fontsize=8, color=C_FAKE if y[i] else C_REAL)
        fig.suptitle(f"tag: {k} ({t[k].sum()} images, random sample)", fontsize=10, x=0.01, ha="left")
        save(fig, f"tag_{k}.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    xs = np.arange(len(tags))
    ax[0].bar(xs - 0.2, [t[k][y == 0].mean() for k in tags], 0.4, color=C_REAL, label="real")
    ax[0].bar(xs + 0.2, [t[k][y == 1].mean() for k in tags], 0.4, color=C_FAKE, label="fake")
    ax[0].set_xticks(xs, tags, rotation=25, ha="right"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_ylabel("share of class", fontsize=8, color=MUTED)
    style(ax[0], "Degradation prevalence by class (heuristic tags)")
    co = np.array([[(t[a] & t[b]).sum() for b in tags[:-1]] for a in tags[:-1]])
    im = ax[1].imshow(co, cmap="Blues")
    ax[1].set_xticks(range(len(tags) - 1), tags[:-1], rotation=25, ha="right")
    ax[1].set_yticks(range(len(tags) - 1), tags[:-1])
    for i in range(len(co)):
        for j in range(len(co)):
            ax[1].text(j, i, co[i, j], ha="center", va="center", fontsize=7,
                       color="white" if co[i, j] > co.max() / 2 else INK)
    style(ax[1], "Tag co-occurrence (counts)")
    save(fig, "degradations.png")

    # ---------------- 4. spatial: class mean images
    f32 = lambda m: np.asarray(imgs[m], dtype=np.float64)
    sums = {l: np.zeros((224, 224, 3)) for l in (0, 1)}
    for l in (0, 1):
        for i in np.flatnonzero(y == l):
            sums[l] += imgs[i]
        sums[l] /= (y == l).sum()
    diff = (sums[1] - sums[0]).mean(axis=2)
    N["mean_image_absdiff_max"] = float(np.abs(diff).max())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))
    ax[0].imshow(sums[0].astype(np.uint8)); ax[0].set_title("mean REAL", fontsize=9)
    ax[1].imshow(sums[1].astype(np.uint8)); ax[1].set_title("mean FAKE", fontsize=9)
    v = np.abs(diff).max()
    im = ax[2].imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v); ax[2].set_title("FAKE − REAL (luma levels)", fontsize=9)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    save(fig, "mean_images.png")

    # ---------------- 5. frequency
    fy, fx = np.meshgrid(np.fft.fftfreq(224), np.fft.fftfreq(224), indexing="ij")
    rad = np.minimum((np.sqrt(fx ** 2 + fy ** 2) / 0.5 * 64).astype(int), 64)
    spec_sum = {0: np.zeros((224, 224)), 1: np.zeros((224, 224))}
    radial = np.zeros((len(y), 65))
    cnt = np.bincount(rad.ravel(), minlength=65)
    for i in range(len(y)):
        g = np.asarray(imgs[i], dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        ls = np.log(np.abs(np.fft.fft2(g - g.mean())) ** 2 + 1)
        spec_sum[y[i]] += ls
        radial[i] = np.bincount(rad.ravel(), weights=ls.ravel(), minlength=65) / np.maximum(cnt, 1)
    for l in (0, 1):
        spec_sum[l] = np.fft.fftshift(spec_sum[l] / (y == l).sum())
    r_auc = np.array([roc_auc_score(y, radial[:, k]) for k in range(65)])
    N["radial_best_band"] = dict(band=int(np.argmax(np.abs(r_auc - 0.5))), auc=float(r_auc[np.argmax(np.abs(r_auc - 0.5))]))
    N["lr_radial_auc"] = grouped_lr_auc(radial, y, folds)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    sd = spec_sum[1] - spec_sum[0]; v = np.abs(sd).max()
    im = ax[0].imshow(sd, cmap="RdBu_r", vmin=-v, vmax=v); ax[0].axis("off")
    ax[0].set_title("mean log power spectrum, FAKE − REAL\n(centre = low frequency)", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    fr = np.arange(65) / 64
    for l, c, n in ((0, C_REAL, "real"), (1, C_FAKE, "fake")):
        m = radial[y == l].mean(0); se = radial[y == l].std(0) / np.sqrt((y == l).sum())
        ax[1].plot(fr, m, color=c, lw=1.8, label=n); ax[1].fill_between(fr, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.25)
    ax[1].set_xlabel("radial frequency (fraction of Nyquist)", fontsize=8, color=MUTED)
    ax[1].set_ylabel("mean log power", fontsize=8, color=MUTED); ax[1].legend(fontsize=8, frameon=False)
    style(ax[1], "Azimuthally averaged spectrum (95% CI bands)")
    ax[2].plot(fr, r_auc, color=INK, lw=1.5); ax[2].axhline(0.5, color=MUTED, ls=":")
    ax[2].set_xlabel("radial frequency (fraction of Nyquist)", fontsize=8, color=MUTED)
    ax[2].set_ylabel("AUC (power in band → fake)", fontsize=8, color=MUTED)
    style(ax[2], "Per-band separability")
    save(fig, "frequency.png")

    # ---------------- 6. near-duplicate structure
    cl = d.groupby("cluster_id").agg(size=("label", "size"), fakes=("label", "sum"))
    cl["mixed"] = (cl.fakes > 0) & (cl.fakes < cl["size"])
    N["clusters"] = dict(n=len(cl), sizes=cl["size"].value_counts().sort_index().to_dict(),
                         multi=int((cl["size"] > 1).sum()), images_in_multi=int(cl["size"][cl["size"] > 1].sum()),
                         mixed=int(cl.mixed.sum()), images_in_mixed=int(cl["size"][cl.mixed].sum()))
    mixed_ids = cl.index[cl.mixed][:6]
    fig, ax = plt.subplots(len(mixed_ids), 6, figsize=(11, 1.9 * len(mixed_ids)))
    for r, cid in enumerate(mixed_ids):
        idx = np.flatnonzero(d.cluster_id.values == cid)[:6]
        for j in range(6):
            ax[r, j].axis("off")
            if j < len(idx):
                ax[r, j].imshow(imgs[idx[j]])
                ax[r, j].set_title("FAKE" if y[idx[j]] else "REAL", fontsize=7.5, color=C_FAKE if y[idx[j]] else C_REAL)
    fig.suptitle("Near-duplicate clusters that contain both labels (first 6)", fontsize=10, x=0.01, ha="left")
    save(fig, "mixed_clusters.png")

    # ---------------- 7. label noise (confident learning flags)
    li = pd.read_csv(ROOT / "data" / "label_issues.csv").set_index("image_id").loc[d.image_id]
    flag = li.flagged.values.astype(bool)
    N["label_issues"] = dict(n=int(flag.sum()), share=round(float(flag.mean()), 4),
                             real=int(flag[y == 0].sum()), fake=int(flag[y == 1].sum()),
                             by_tag={k: round(float(flag[t[k].values].mean()), 4) for k in tags})

    # ---------------- 8. embedding view
    fe = np.load(ROOT / "experiments" / "cache" / "feats" / "convnext_t" / "clean_last.npy").astype(np.float32)
    emb = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(PCA(50, random_state=0).fit_transform(fe))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for a, (col, title) in zip(ax, ((None, "coloured by label"), ("tag", "coloured by detected degradation"))):
        if col is None:
            for l, c, n in ((0, C_REAL, "real"), (1, C_FAKE, "fake")):
                a.scatter(*emb[y == l].T, s=2, c=c, alpha=0.5, label=n, lw=0)
            a.legend(fontsize=8, frameon=False, markerscale=4)
        else:
            primary = np.where(t.none_detected, "none_detected", t.drop(columns="none_detected").idxmax(axis=1))
            cmap = plt.get_cmap("tab10")
            for j, k in enumerate(tags):
                m = primary == k
                a.scatter(*emb[m].T, s=2, color=cmap(j), alpha=0.6, label=k, lw=0)
            a.legend(fontsize=7, frameon=False, markerscale=4, loc="best")
        a.set_xticks([]); a.set_yticks([]); style(a, f"t-SNE of ImageNet ConvNeXt-T features, {title}")
    save(fig, "tsne.png")

    # ---------------- 9. where the shipped model fails (out-of-fold ensemble)
    p = ensemble_oof(d)
    ok = ~np.isnan(p)
    N["oof_auc"] = float(roc_auc_score(y[ok], p[ok]))
    per_tag = {}
    for k in tags:
        m = ok & t[k].values
        if m.sum() > 30 and len(np.unique(y[m])) == 2:
            per_tag[k] = dict(n=int(m.sum()), auc=round(float(roc_auc_score(y[m], p[m])), 4),
                              acc=round(float(((p[m] >= 0.5) == y[m]).mean()), 4))
    N["oof_by_tag"] = per_tag
    per_q = []
    for c in ("brightness", "contrast", "sharpness", "noise"):
        q = pd.qcut(d[c], 5, labels=False, duplicates="drop")
        per_q.append({"feature": c, **{f"Q{j + 1}": round(float(roc_auc_score(y[ok & (q == j)], p[ok & (q == j)])), 3)
                                         for j in range(int(q.max()) + 1)}})
    N["oof_by_quintile"] = per_q
    N["oof_flagged_vs_clean_acc"] = dict(clean=round(float(((p >= 0.5) == y)[ok & ~flag].mean()), 4),
                                         flagged=round(float(((p >= 0.5) == y)[ok & flag].mean()), 4))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ks = list(per_tag)
    ax[0].barh(range(len(ks)), [per_tag[k]["auc"] for k in ks], color=INK)
    ax[0].axvline(N["oof_auc"], color=C_FAKE, ls="--", lw=1.2, label=f"overall {N['oof_auc']:.3f}")
    ax[0].set_yticks(range(len(ks)), [f"{k} (n={per_tag[k]['n']})" for k in ks])
    ax[0].set_xlim(0.75, 0.97); ax[0].legend(fontsize=8, frameon=True, loc="lower left")
    style(ax[0], "Out-of-fold ensemble AUC by degradation tag")
    bins = np.linspace(0, 1, 41)
    ax[1].hist(p[ok & (y == 0)], bins, color=C_REAL, alpha=0.6, label="real")
    ax[1].hist(p[ok & (y == 1)], bins, color=C_FAKE, alpha=0.6, label="fake")
    ax[1].axvline(0.5, color=MUTED, ls=":"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_xlabel("P(fake)", fontsize=8, color=MUTED); style(ax[1], "Out-of-fold score distribution")
    e = np.linspace(0, 1, 11); mid = (e[1:] + e[:-1]) / 2
    b = np.clip(np.digitize(p[ok], e) - 1, 0, 9)
    frac = [y[ok][b == j].mean() if (b == j).any() else np.nan for j in range(10)]
    ax[2].plot([0, 1], [0, 1], color=MUTED, ls=":"); ax[2].plot(mid, frac, "-o", color=INK, ms=5)
    ax[2].set_xlabel("predicted P(fake)", fontsize=8, color=MUTED); ax[2].set_ylabel("observed fake rate", fontsize=8, color=MUTED)
    style(ax[2], "Calibration (reliability curve)")
    def ece(q):
        bb = np.clip(np.digitize(q, e) - 1, 0, 9)
        return float(sum(abs(y[ok][bb == j].mean() - q[bb == j].mean()) * (bb == j).mean() for j in range(10) if (bb == j).any()))

    def nll(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-np.mean(y[ok] * np.log(q) + (1 - y[ok]) * np.log(1 - q)))

    # temperature scaling, cross-validated over the grouped folds (fit T on 4 folds, apply to the 5th)
    zl = np.log(p[ok] / (1 - p[ok]))
    grid = np.linspace(0.5, 6, 111)
    pt = np.zeros_like(zl)
    Ts = []
    for f in np.unique(folds[ok]):
        tr, va = folds[ok] != f, folds[ok] == f
        losses = [-np.mean(y[ok][tr] * np.log(1 / (1 + np.exp(-s * zl[tr]))) + (1 - y[ok][tr]) * np.log(1 - 1 / (1 + np.exp(-s * zl[tr])) + 1e-9)) for s in grid]
        s = grid[int(np.argmin(losses))]; Ts.append(float(s))
        pt[va] = 1 / (1 + np.exp(-s * zl[va]))
    N["calibration"] = dict(ece_before=ece(p[ok]), ece_after=ece(pt), nll_before=nll(p[ok]), nll_after=nll(pt),
                            logit_scale_per_fold=Ts, acc_unchanged=bool(((pt >= 0.5) == (p[ok] >= 0.5)).all()))
    N["ece"] = N["calibration"]["ece_before"]
    bt = np.clip(np.digitize(pt, e) - 1, 0, 9)
    ax[2].plot(mid, [y[ok][bt == j].mean() if (bt == j).any() else np.nan for j in range(10)], "-s", color=C_FAKE, ms=4,
               label=f"after logit × {np.mean(Ts):.1f} (CV)")
    ax[2].lines[1].set_label("shipped"); ax[2].legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "model_errors.png")

    wrong = np.flatnonzero(ok & ((p >= 0.5) != y))
    worst = wrong[np.argsort(-np.abs(p[wrong] - 0.5))][:12]
    fig, ax = plt.subplots(2, 6, figsize=(12, 5.4))
    for a, i in zip(ax.ravel(), worst):
        a.imshow(imgs[i]); a.axis("off")
        a.set_title(f"label {'FAKE' if y[i] else 'REAL'}\nP(fake)={p[i]:.2f}{' *' if flag[i] else ''}", fontsize=7.5,
                    color=C_FAKE if y[i] else C_REAL)
    fig.suptitle("Most confident out-of-fold errors (* = flagged as a likely label error by cleanlab)", fontsize=10, x=0.01, ha="left")
    save(fig, "confident_errors.png")
    N["confident_errors_flagged"] = int(flag[worst].sum())

    # ---------------- 10. random samples
    fig, ax = plt.subplots(2, 8, figsize=(14, 4))
    for r, l in enumerate((0, 1)):
        for a, i in zip(ax[r], rng.choice(np.flatnonzero(y == l), 8, replace=False)):
            a.imshow(imgs[i]); a.axis("off")
        ax[r, 0].set_title("REAL" if l == 0 else "FAKE", loc="left", fontsize=10, color=C_FAKE if l else C_REAL)
    save(fig, "samples.png")

    (EDA / "eda_numbers.json").write_text(json.dumps(N, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(N, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
