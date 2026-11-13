"""T1.6 robustness proxy: fixed, seeded perturbations of validation images.

Each perturbation maps uint8 HxWx3 RGB -> uint8 HxWx3 RGB and is deterministic
given (image index, perturbation name). Applied to validation images only.
"""
import zlib

import cv2
import numpy as np


def _rng(i: int, name: str) -> np.random.Generator:
    return np.random.default_rng([i, zlib.crc32(name.encode())])


def jpeg(q):
    def f(im, i):
        ok, buf = cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return f


def blur(sigma):
    return lambda im, i: cv2.GaussianBlur(im, (0, 0), sigma)


def down_up(scale):
    def f(im, i):
        h, w = im.shape[:2]
        small = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return f


def noise(std):
    def f(im, i):
        n = _rng(i, f"noise{std}").normal(0, std, im.shape)
        return np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return f


def colour_jitter(im, i):
    x = im.astype(np.float32)
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * 0.8                     # saturation
    x = (x - x.mean()) * 0.85 + x.mean() * 1.10     # contrast, brightness
    return np.clip(x, 0, 255).astype(np.uint8)


def occlusion(size=40):
    def f(im, i):
        r = _rng(i, "occ")
        h, w = im.shape[:2]
        y, x = r.integers(0, h - size), r.integers(0, w - size)
        out = im.copy()
        out[y:y + size, x:x + size] = 128
        return out
    return f


def grayscale(im, i):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)


def pixelate(scale):
    def f(im, i):
        h, w = im.shape[:2]
        small = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return f


def gamma(g):
    lut = (255 * (np.arange(256) / 255) ** g).astype(np.uint8)
    return lambda im, i: lut[im]


PERTURBATIONS = {
    "jpeg75": jpeg(75),
    "jpeg50": jpeg(50),
    "blur1": blur(1.0),
    "blur2.5": blur(2.5),
    "downup0.5": down_up(0.5),
    "noise8": noise(8),
    "colour": colour_jitter,
    "occlude40": occlusion(40),
    "gray": grayscale,
    "pixelate0.25": pixelate(0.25),
    "gamma0.6": gamma(0.6),
}
# Families used by training augmentation (src/augment.py) vs never trained on.
SEEN = ["jpeg75", "jpeg50", "blur1", "blur2.5", "downup0.5", "colour"]
UNSEEN = ["noise8", "occlude40", "gray", "pixelate0.25", "gamma0.6"]
ORIGINAL8 = ["jpeg75", "jpeg50", "blur1", "blur2.5", "downup0.5", "noise8", "colour", "occlude40"]


def perturb_all(imgs, idx, name: str) -> np.ndarray:
    f = PERTURBATIONS[name]
    return np.stack([f(np.asarray(imgs[k]), int(k)) for k in idx])


def summarise(clean_auc: float, variant_aucs: dict) -> dict:
    v = list(variant_aucs.values())
    out = {"clean_auc": clean_auc, "robust_mean_auc": float(np.mean(v)),
           "robust_worst_auc": float(np.min(v)),
           "worst_variant": min(variant_aucs, key=variant_aucs.get)}
    for name, keys in (("seen", SEEN), ("unseen", UNSEEN), ("orig8", ORIGINAL8)):
        ks = [k for k in keys if k in variant_aucs]
        if ks:
            out[f"{name}_mean_auc"] = float(np.mean([variant_aucs[k] for k in ks]))
    out.update({f"auc_{k}": a for k, a in variant_aucs.items()})
    return out
