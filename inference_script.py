"""
inference_script.py — Deepfake ML Hackathon (based on the official template)

    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

Model: ensemble of ViT-B/16 and ConvNeXt-T (ImageNet-pretrained via timm, fine-tuned on the provided
training data only), 3 seeds each, horizontal-flip TTA.
Per member: logits averaged over seeds and flip, standardised with mean/std fitted on 5-fold
out-of-fold training predictions (stored inside the weights file), then weighted sum (0.7 / 0.3).
confidence = sigmoid(ensemble score) = P(fake); label = confidence >= threshold.
Nothing is fitted on the test images. No hardcoded paths. No internet access needed.

The weights file bundles every member's state dict and the ensemble config (see scripts/pack_weights.py).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 224   # training images (and test images) are 224x224
BATCH_SIZE = 64
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — flat directory; identical preprocessing to training (src/data.py: load_rgb)
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, image_dir: str):
        d = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        try:  # grayscale / RGBA / CMYK / palette -> RGB
            img = Image.open(p).convert("RGB")
            if img.size != (IMAGE_SIZE, IMAGE_SIZE):
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            x, ok = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1), True
        except Exception as e:  # unreadable file: still one row, neutral confidence
            print(f"[WARN] could not read {p.name}: {e}; writing confidence 0.5")
            x, ok = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8), False
        return x, p.stem, ok  # image_id = filename without extension


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class Member:
    """One architecture: several seed models sharing normalisation and z-logit stats."""

    def __init__(self, cfg: dict, states: dict):
        self.weight = float(cfg["weight"])
        self.zm, self.zs = float(cfg["zlogit_mean"]), float(cfg["zlogit_std"])
        self.mean = torch.tensor(cfg["norm_mean"], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["norm_std"], device=DEVICE).view(1, 3, 1, 1)
        self.models = []
        for f in cfg["files"]:
            m = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=1)
            m.load_state_dict({k: v.float() for k, v in states[f].items()}, strict=True)
            self.models.append(m.to(DEVICE).eval())

    @torch.inference_mode()
    def score(self, x01: torch.Tensor, tta_flip: bool, amp: bool) -> torch.Tensor:
        x = (x01 - self.mean) / self.std
        views = [x, x.flip(3)] if tta_flip else [x]
        logits = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for m in self.models:
                for v in views:
                    logits.append(m(v).float().squeeze(1))
        z = torch.stack(logits).mean(0)
        return self.weight * (z - self.zm) / self.zs


def load_model(weights_path: str):
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    assert isinstance(blob, dict) and blob.get("format") == "deepfake-ensemble-v1", "unexpected weights file"
    cfg = blob["config"]
    members = [Member(c, blob["state_dicts"]) for c in cfg["members"].values()]
    n = sum(len(m.models) for m in members)
    print(f"[INFO] Loaded {n} models ({', '.join(cfg['members'])}) from {weights_path} on {DEVICE}")
    return members, cfg


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(members, cfg, test_dir: str, batch_size: int, threshold: float, amp: bool) -> pd.DataFrame:
    dataset = TestDataset(test_dir)
    workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=DEVICE == "cuda", persistent_workers=False)
    print(f"[INFO] Running inference on {len(dataset)} images (batch_size={batch_size}, workers={workers}, "
          f"tta_flip={cfg['tta_flip']}, fp16={amp})...")
    ids, scores = [], []
    for xb, idb, okb in loader:
        x01 = xb.to(DEVICE, non_blocking=True).float().div_(255)
        s = sum(m.score(x01, cfg["tta_flip"], amp) for m in members).cpu().numpy().astype(np.float64)
        s[~okb.numpy().astype(bool)] = 0.0  # unreadable -> sigmoid(0) = 0.5
        scores.append(s)
        ids.extend(idb)
    conf = 1 / (1 + np.exp(-np.concatenate(scores)))
    return pd.DataFrame({"image_id": ids, "label": (conf >= threshold).astype(int), "confidence": conf})


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images directory")
    parser.add_argument("--weights", required=True, help="Path to model weights file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--fp32", action="store_true",
                        help="full precision (slower; default is fp16 autocast, labels agree 99.97%% with fp32)")
    args = parser.parse_args()

    t0 = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    members, cfg = load_model(args.weights)
    predictions = run_inference(members, cfg, args.test_dir, args.batch_size, args.threshold,
                                amp=(not args.fp32) and DEVICE == "cuda")

    assert len(predictions) > 0, "no images found in --test_dir"
    assert predictions["image_id"].is_unique, "duplicate image_id"
    assert predictions["confidence"].between(0, 1).all(), "Confidence values must be in [0, 1]"
    assert predictions["label"].isin([0, 1]).all(), "Labels must be 0 or 1"

    predictions.to_csv(args.output, index=False)
    print(f"[DONE] Predictions saved to {args.output} in {time.time() - t0:.1f}s")
    print(f"  Total images: {len(predictions)}")
    print(f"  Predicted FAKE: {predictions['label'].sum()} ({predictions['label'].mean() * 100:.1f}%)")
    print(f"  Predicted REAL: {(predictions['label'] == 0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
