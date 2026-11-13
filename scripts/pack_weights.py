"""Bundle weights/*.pt + weights/ensemble.json into ONE file for `--weights` (the CLI takes a single path).

Usage: python scripts/pack_weights.py [--src weights] [--out model_weights.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "weights"))
    ap.add_argument("--out", default=str(ROOT / "model_weights.pt"))
    a = ap.parse_args()
    src = Path(a.src)
    cfg = json.loads((src / "ensemble.json").read_text())
    states = {f: torch.load(src / f, map_location="cpu")
              for m in cfg["members"].values() for f in m["files"]}
    torch.save({"format": "deepfake-ensemble-v1", "config": cfg, "state_dicts": states}, a.out)
    print(f"packed {len(states)} models -> {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
