"""Hyperparameter evidence figure for the report, read from experiments/results.csv (no typed-in numbers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import ROOT

C1, C2 = "#2a78d6", "#eb6834"  # categorical slots 1-2 (blue = ViT / first series, orange = second)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

r = pd.read_csv(ROOT / "experiments" / "results.csv")
r["name"] = r.run_id.str.split("_", n=1).str[1]
R = r.drop_duplicates("name", keep="last").set_index("name")


def g(n, col="cv_auc"):
    return float(R.loc[n, col])


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
fig.patch.set_facecolor("#fcfcfb")

# (a) frozen probe: last vs intermediate layers
a = ax[0, 0]
bb = [("vit_b16", "ViT-B/16"), ("convnext_t", "ConvNeXt-T"), ("resnet50", "ResNet-50"), ("effnet_b3", "EffNet-B3")]
x = np.arange(len(bb))
last = [g(f"probe_{k}_last") for k, _ in bb]
mid = [g(f"probe_{k}_mid") for k, _ in bb]
a.bar(x - 0.19, last, 0.36, color=C1, label="last layer", edgecolor="#fcfcfb", lw=2)
a.bar(x + 0.19, mid, 0.36, color=C2, label="pooled intermediate layers", edgecolor="#fcfcfb", lw=2)
for i, v in enumerate(mid):
    a.text(i + 0.19, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
a.set_xticks(x, [n for _, n in bb])
a.set_ylim(0.70, 0.88); a.set_ylabel("grouped 5-fold AUC", fontsize=8, color=MUTED)
a.axhline(g("baseline"), color=MUTED, ls=":", lw=1)
a.text(-0.45, g("baseline") + 0.003, "pixel baseline", fontsize=7, color=MUTED, ha="left")
a.legend(fontsize=7.5, frameon=False, loc="upper right")
style(a, "(a) Day 2: frozen-feature probe (logistic regression)")

# (b) tuning regime, fold 0
a = ax[0, 1]
reg = [("ln", "LayerNorm only"), ("partial", "top blocks"), ("full", "full fine-tune")]
vit = [g("d3_vit_ln_lr5e-4_e8"), g("d3_vit_partial_lr5e-5_e5"), g("d3_vit_full_lr5e-5_e4")]
cnx = [g("d3_cnx_ln_lr1e-3_e8"), g("d3_cnx_partial_lr1e-4_e5"), g("d3_cnx_full_lr1e-4_e4")]
x = np.arange(3)
a.plot(x, vit, "-o", color=C1, lw=2, ms=8, label="ViT-B/16")
a.plot(x, cnx, "-o", color=C2, lw=2, ms=8, label="ConvNeXt-T")
a.set_xticks(x, [n for _, n in reg]); a.set_ylim(0.83, 0.92)
a.set_ylabel("fold-0 AUC", fontsize=8, color=MUTED)
a.text(2.08, vit[2], "ViT-B/16", fontsize=8, color=INK, va="center")
a.text(2.08, cnx[2] - 0.003, "ConvNeXt-T", fontsize=8, color=INK, va="center")
a.set_xlim(-0.2, 2.6)
style(a, "(b) Day 3: what to unfreeze (fold 0)")

# (c) degradation augmentation strength
a = ax[1, 0]
lv = ["off", "weak", "medium", "strong", "xstrong"]
cx = {l: f"d4_cnx_deg_{l}" for l in lv}
vx = {"off": "d4_vit_deg_off", "strong": "d4_vit_deg_strong", "xstrong": "d4_vit_deg_xstrong"}
xi = {l: i for i, l in enumerate(lv)}
a.plot([xi[l] for l in lv], [g(cx[l]) for l in lv], "-o", color=C2, lw=2, ms=7, label="ConvNeXt-T clean")
a.plot([xi[l] for l in lv], [g(cx[l], "robust_worst_auc") for l in lv], "--o", color=C2, lw=2, ms=7,
       mfc="#fcfcfb", label="ConvNeXt-T worst perturbation")
a.plot([xi[l] for l in vx], [g(vx[l]) for l in vx], "-o", color=C1, lw=2, ms=7, label="ViT-B/16 clean")
a.plot([xi[l] for l in vx], [g(vx[l], "robust_worst_auc") for l in vx], "--o", color=C1, lw=2, ms=7,
       mfc="#fcfcfb", label="ViT-B/16 worst perturbation")
a.set_xticks(range(5), lv); a.set_ylim(0.80, 0.92)
a.set_ylabel("folds 0-1 AUC", fontsize=8, color=MUTED)
a.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)
style(a, "(c) Day 4: degradation-augmentation strength")

# (d) label-noise handling (ConvNeXt-T, folds 0-1), change vs BCE baseline
a = ax[1, 1]
base = "d4_cnx_deg_xstrong"
var = [("d5_cnx_ls0.1", "label smoothing 0.1"), ("d5_cnx_ema", "EMA 0.99"), ("d5_cnx_sce", "SCE loss"),
       ("d5_cnx_gce", "GCE loss"), ("d5_cnx_prune", "prune cleanlab flags")]
y = np.arange(len(var))
dc = [g(n) - g(base) for n, _ in var]
dw = [g(n, "robust_worst_auc") - g(base, "robust_worst_auc") for n, _ in var]
a.barh(y + 0.19, dc, 0.36, color=C1, label="clean AUC", edgecolor="#fcfcfb", lw=2, zorder=3)
a.barh(y - 0.19, dw, 0.36, color=C2, label="worst-perturbation AUC", edgecolor="#fcfcfb", lw=2, zorder=3)
a.axvline(0, color=MUTED, lw=1)
a.axvspan(-0.012, 0.012, color="#f0efec", lw=0, zorder=0); a.set_xlim(-0.02, 0.02)
a.set_yticks(y, [n for _, n in var]); a.xaxis.grid(True, color=GRID); a.yaxis.grid(False)
a.set_xlabel("change vs BCE baseline (ConvNeXt-T, folds 0-1)", fontsize=8, color=MUTED)
a.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.62))
style(a, "(d) Day 5: label-noise handling")
a.yaxis.grid(False)

fig.tight_layout()
out = ROOT / "reports" / "figures" / "hyperparameter_evidence.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print("saved", out)
