"""T6.4 Grad-CAM for both ensemble members (bonus +2).

Uses one final model per member (seed 1000) on training images (test images are never used).
Target: the evidence for the predicted class (logit for FAKE, -logit for REAL).
Sanity check (Adebayo et al.): maps from a randomly initialised network must differ from the trained ones.
Output: reports/figures/gradcam.png, printed sanity correlations.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.common import ROOT
from src.data import train_images


class Logit:
    def __init__(self, sign):
        self.sign = sign

    def __call__(self, out):
        return self.sign * out.reshape(-1)[0]


def vit_reshape(t, h=14, w=14):
    return t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


def load(member, cfg, random_init=False):
    c = cfg["members"][member]
    m = timm.create_model(c["timm_name"], pretrained=False, num_classes=1)
    if not random_init:
        sd = torch.load(ROOT / "weights" / c["files"][0], map_location="cpu")
        m.load_state_dict({k: v.float() for k, v in sd.items()})
    m = m.cuda().eval()
    if member == "vit":
        cam = GradCAM(model=m, target_layers=[m.blocks[-1].norm1], reshape_transform=vit_reshape)
    else:
        cam = GradCAM(model=m, target_layers=[m.stages[-1]])
    mean = torch.tensor(c["norm_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(c["norm_std"]).view(1, 3, 1, 1)
    return m, cam, mean, std


def maps(member, cfg, imgs_u8, random_init=False):
    m, cam, mean, std = load(member, cfg, random_init)
    x = ((torch.from_numpy(imgs_u8).permute(0, 3, 1, 2).float() / 255 - mean) / std).cuda()
    with torch.no_grad():
        logit = m(x).float().squeeze(1).cpu().numpy()
    out = np.stack([cam(input_tensor=x[i:i + 1], targets=[Logit(1 if logit[i] > 0 else -1)])[0]
                    for i in range(len(x))])
    del m, cam
    torch.cuda.empty_cache()
    return out, logit


def main():
    cfg = json.loads((ROOT / "weights" / "ensemble.json").read_text())
    df, imgs = train_images()
    li = pd.read_csv(ROOT / "data" / "label_issues.csv").set_index("image_id")
    clean = ~df.image_id.map(li.flagged).fillna(False).values.astype(bool)
    rng = np.random.default_rng(7)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(clean & (df.label.values == l)), 4, replace=False)
                                  for l in (0, 1)]))
    x = np.asarray(imgs[idx])
    y = df.label.values[idx]

    res = {m: maps(m, cfg, x) for m in ("vit", "cnx")}
    rnd = {m: maps(m, cfg, x, random_init=True)[0] for m in ("vit", "cnx")}
    for m in res:
        r = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(res[m][0], rnd[m])]
        print(f"sanity {m}: corr(trained map, random-init map) mean {np.nanmean(r):.3f} (should be low)")

    fig, ax = plt.subplots(3, len(idx), figsize=(2.1 * len(idx), 6.6))
    for j in range(len(idx)):
        rgb = x[j].astype(np.float32) / 255
        ax[0, j].imshow(x[j])
        ax[0, j].set_title(f"{'FAKE' if y[j] else 'REAL'}", color="crimson" if y[j] else "seagreen", fontsize=10)
        for r, m in ((1, "vit"), (2, "cnx")):
            p = 1 / (1 + np.exp(-res[m][1][j]))
            ax[r, j].imshow(show_cam_on_image(rgb, res[m][0][j], use_rgb=True))
            ax[r, j].set_title(f"P(fake)={p:.2f}", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("image"); ax[1, 0].set_ylabel("ViT-B/16"); ax[2, 0].set_ylabel("ConvNeXt-T")
    fig.suptitle("Grad-CAM: evidence for the predicted class (final models, training images)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "gradcam.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
