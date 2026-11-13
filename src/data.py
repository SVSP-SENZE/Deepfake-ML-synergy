"""Training data loading. Images are decoded once and cached as a uint8 array."""
import numpy as np
import pandas as pd
from PIL import Image

from .common import CACHE, DATA_ROOT, FOLDS_CSV

LABEL_DIR = {0: "real", 1: "fake"}


def train_table() -> pd.DataFrame:
    """train_metadata.csv plus the file path; checks every file exists."""
    df = pd.read_csv(DATA_ROOT / "train_metadata.csv")
    df["path"] = [str(DATA_ROOT / "train" / LABEL_DIR[l] / f"{i}.jpg")
                  for i, l in zip(df.image_id, df.label)]
    return df.sort_values("image_id").reset_index(drop=True)


def load_rgb(path: str, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def train_images(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (table, images[N,224,224,3] uint8), in image_id order. Cached on disk."""
    df = train_table() if df is None else df
    CACHE.mkdir(parents=True, exist_ok=True)
    arr_p, ids_p = CACHE / "train_u8.npy", CACHE / "train_ids.txt"
    if arr_p.exists() and ids_p.exists() and ids_p.read_text().split() == list(df.image_id):
        return df, np.load(arr_p, mmap_mode="r")
    imgs = np.stack([load_rgb(p) for p in df.path])
    np.save(arr_p, imgs)
    ids_p.write_text("\n".join(df.image_id))
    return df, imgs


def with_folds(df: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(FOLDS_CSV)
    out = df.merge(folds[["image_id", "cluster_id", "fold"]], on="image_id", how="left")
    assert out.fold.notna().all(), "folds.csv does not cover every training image"
    return out
