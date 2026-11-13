"""T3.1 degradation augmentation (CPU, cv2, on uint8 batches before GPU upload).

Applied identically to both classes. Each op fires independently with probability p.
Order mimics a real pipeline: colour -> down-up resample -> blur -> JPEG.
Noise and occlusion are deliberately NOT here, so the robustness proxy keeps
"unseen" perturbation types (see src/robustness.py: SEEN / UNSEEN).
"""
import cv2
import numpy as np

LEVELS = {
    "off": None,
    "weak":   {"p": 0.2, "jpeg": (70, 95), "sigma": (0.5, 1.2), "scale": (0.7, 0.9), "colour": 0.10},
    "medium": {"p": 0.3, "jpeg": (50, 95), "sigma": (0.5, 2.0), "scale": (0.5, 0.9), "colour": 0.15},
    "strong": {"p": 0.5, "jpeg": (30, 95), "sigma": (0.5, 2.5), "scale": (0.5, 0.9), "colour": 0.20},
    "xstrong": {"p": 0.7, "jpeg": (20, 95), "sigma": (0.5, 3.0), "scale": (0.35, 0.9), "colour": 0.25},
}


def _colour(im, amt, rng):
    x = im.astype(np.float32)
    gray = x.mean(axis=2, keepdims=True)
    sat = 1 + rng.uniform(-amt, amt)
    con = 1 + rng.uniform(-amt, amt)
    bri = 1 + rng.uniform(-amt, amt)
    x = gray + (x - gray) * sat
    x = (x - x.mean()) * con + x.mean() * bri
    return np.clip(x, 0, 255).astype(np.uint8)


def degrade(im: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    p = cfg["p"]
    h, w = im.shape[:2]
    if rng.random() < p:
        im = _colour(im, cfg["colour"], rng)
    if rng.random() < p:
        s = rng.uniform(*cfg["scale"])
        small = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (w, h), interpolation=[cv2.INTER_LINEAR, cv2.INTER_CUBIC][rng.integers(2)])
    if rng.random() < p:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(*cfg["sigma"]))
    if rng.random() < p:
        q = int(rng.integers(cfg["jpeg"][0], cfg["jpeg"][1] + 1))
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(im)


def degrade_batch(batch: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    cfg = LEVELS[level]
    if cfg is None:
        return batch
    return np.stack([degrade(np.asarray(im), cfg, rng) for im in batch])
