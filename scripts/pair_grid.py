"""Save a review grid for candidate pairs matching a pandas query.
Usage: python scripts/pair_grid.py "c16 >= 0.90 and c16 < 0.93" name [n]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common import ROOT
from src.data import train_images


def grid(pairs: pd.DataFrame, out: Path, n: int = 12, seed: int = 0) -> None:
    df, imgs = train_images()
    idx = {v: k for k, v in enumerate(df.image_id)}
    take = pairs.iloc[np.random.default_rng(seed).permutation(len(pairs))[:n]]
    tile, name = 160, {0: "REAL", 1: "FAKE"}
    canvas = Image.new("RGB", (tile * 4 + 30, tile * ((len(take) + 1) // 2) + 10), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(take.itertuples()):
        x0, y0 = (k % 2) * (tile * 2 + 20), (k // 2) * tile
        for m, (iid, l) in enumerate([(r.a, r.label_a), (r.b, r.label_b)]):
            im = Image.fromarray(np.asarray(imgs[idx[iid]])).resize((tile - 4, tile - 18))
            canvas.paste(im, (x0 + m * tile, y0 + 16))
            d.text((x0 + m * tile + 2, y0 + 2), name[l], fill="red" if l else "green")
        d.text((x0 + 44, y0 + 2), f"c16={r.c16:.3f} c64={r.c64:.2f} h={r.ham}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


if __name__ == "__main__":
    q, name = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    p = pd.read_csv(ROOT / "data" / "dup_candidates.csv").query(q)
    print(f"{len(p)} pairs match: {q}")
    grid(p, ROOT / "reports" / "figures" / "dups" / f"q_{name}.png", n)
