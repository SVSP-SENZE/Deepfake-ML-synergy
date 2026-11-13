"""T2.2 step 1: cache frozen backbone features for all train images, clean and perturbed.

Output: experiments/cache/feats/<backbone>/<variant>_{last,mid}.npy  (float16)
Usage: python scripts/extract_features.py convnext_t [vit_b16 ...] [--clean-only]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.backbones import build, features, norm_stats, to_input
from src.common import CACHE, seed_everything
from src.data import train_images
from src.robustness import PERTURBATIONS, perturb_all

BS = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbones", nargs="+")
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()
    seed_everything(0)
    _, imgs = train_images()
    n = len(imgs)
    dev = "cuda"
    variants = ["clean"] + ([] if args.clean_only else list(PERTURBATIONS))
    for key in args.backbones:
        out = CACHE / "feats" / key
        out.mkdir(parents=True, exist_ok=True)
        model = build(key).to(dev).eval()
        mean, std = norm_stats(model)
        for v in variants:
            if (out / f"{v}_mid.npy").exists():
                print(f"{key}/{v}: cached"); continue
            t0 = time.time()
            last, mid = [], []
            for s in range(0, n, BS):
                idx = np.arange(s, min(s + BS, n))
                batch = np.asarray(imgs[idx]) if v == "clean" else perturb_all(imgs, idx, v)
                with torch.autocast("cuda", dtype=torch.float16):
                    l, m = features(model, key, to_input(batch, mean, std, dev))
                last.append(l.cpu().numpy().astype(np.float16))
                mid.append(m.cpu().numpy().astype(np.float16))
            np.save(out / f"{v}_last.npy", np.concatenate(last))
            np.save(out / f"{v}_mid.npy", np.concatenate(mid))
            print(f"{key}/{v}: {time.time() - t0:.0f}s, last {last[0].shape[1]}, mid {mid[0].shape[1]}",
                  flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
