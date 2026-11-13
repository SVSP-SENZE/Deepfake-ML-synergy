"""Fine-tuning on grouped folds (T2.3 / T2.4).

Images come from the memory-mapped uint8 cache; augmentation runs on the GPU
(no DataLoader workers: little host memory on this machine).
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .augment import degrade_batch
from .backbones import build, norm_stats, to_input
from .robustness import PERTURBATIONS, perturb_all


def gpu_augment(x: torch.Tensor, cfg: dict, gen: torch.Generator) -> torch.Tensor:
    """x: float [B,3,H,W] in [0,1]. Light, label-independent augmentation."""
    b = x.shape[0]
    if cfg.get("hflip", True):
        flip = torch.rand(b, generator=gen, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = cfg.get("shift", 0)
    if s:  # small random translation via reflect pad + crop
        h, w = x.shape[2:]
        x = F.pad(x, (s, s, s, s), mode="reflect")
        dx, dy = torch.randint(0, 2 * s + 1, (2,), generator=gen, device=x.device).tolist()
        x = x[:, :, dy:dy + h, dx:dx + w]
    return x


PARTIAL_PREFIXES = {  # top of each network; unfrozen in the "partial" regime
    "vit_b16": tuple(f"blocks.{i}." for i in range(8, 12)) + ("norm.", "fc_norm."),
    "convnext_t": ("stages.3.", "norm_pre."),
}


def set_trainable(model, regime: str, key: str = ""):
    """full | layernorm (norm layers + head) | partial (top blocks + head) | head."""
    for n, p in model.named_parameters():
        if regime == "full":
            p.requires_grad = True
        elif regime == "partial":
            p.requires_grad = n.startswith(PARTIAL_PREFIXES[key] + ("head",))
        elif regime == "layernorm":
            p.requires_grad = ("norm" in n) or ("bn" in n) or n.startswith(("head", "fc", "classifier"))
        elif regime == "head":
            p.requires_grad = n.startswith(("head", "fc", "classifier"))
        else:
            raise ValueError(regime)


def head_params(model):
    names = ("head", "fc.", "classifier")
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(names)]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(names)]
    return head, body


@torch.inference_mode()
def predict(model, imgs, idx, mean, std, dev, bs=128, variant=None, tta_flip=False):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        bi = idx[s:s + bs]
        batch = np.asarray(imgs[bi]) if variant is None else perturb_all(imgs, bi, variant)
        x = to_input(batch, mean, std, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x).float().squeeze(1)
            if tta_flip:
                z = (z + model(x.flip(3)).float().squeeze(1)) / 2
        out.append(z.cpu().numpy())
    return np.concatenate(out)  # logits


def loss_fn(logit: torch.Tensor, t: torch.Tensor, cfg: dict) -> torch.Tensor:
    """bce (optionally label-smoothed) | gce (generalised CE, q) | sce (CE + reverse CE)."""
    kind = cfg.get("loss", "bce")
    if kind == "bce":
        s = cfg.get("label_smoothing", 0.0)
        return F.binary_cross_entropy_with_logits(logit, t * (1 - s) + 0.5 * s)
    p = torch.sigmoid(logit)
    p_t = torch.where(t > 0.5, p, 1 - p).clamp(1e-6, 1)
    if kind == "gce":
        q = cfg.get("gce_q", 0.7)
        return ((1 - p_t ** q) / q).mean()
    if kind == "sce":  # RCE with log(0) clamped to A=-4 reduces to 4*(1-p_t)
        return F.binary_cross_entropy_with_logits(logit, t) + cfg.get("sce_beta", 1.0) * 4 * (1 - p_t).mean()
    raise ValueError(kind)


class EMA:
    def __init__(self, model, decay):
        import copy
        self.decay, self.step = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def train_fold(cfg: dict, imgs, y, tr, va, seed: int, log=print) -> dict:
    dev = "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(cfg["backbone"], num_classes=1).to(dev)
    set_trainable(model, cfg.get("regime", "full"), cfg["backbone"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  regime {cfg.get('regime', 'full')}: {n_train / 1e6:.2f}M trainable params")
    mean, std = norm_stats(model)
    head, body = head_params(model)
    opt = torch.optim.AdamW([{"params": body, "lr": cfg["lr"]},
                             {"params": head, "lr": cfg["lr"] * cfg.get("head_lr_mult", 10)}],
                            weight_decay=cfg.get("wd", 0.05))
    bs, epochs = cfg.get("batch_size", 64), cfg["epochs"]
    steps_per_epoch = len(tr) // bs
    total, warm = steps_per_epoch * epochs, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler()
    ema = EMA(model, cfg["ema"]) if cfg.get("ema") else None
    yt = torch.tensor(y, dtype=torch.float32)
    curve = []
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(tr)
        for s in range(steps_per_epoch):
            bi = np.sort(perm[s * bs:(s + 1) * bs])  # sorted reads are faster from the mmap
            batch = degrade_batch(np.asarray(imgs[bi]), cfg.get("degrade", "off"), rng)
            x = to_input(batch, torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1), dev)
            x = gpu_augment(x, cfg, gen)
            x = (x - mean.to(dev)) / std.to(dev)
            t = yt[bi].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x).float().squeeze(1)
            loss = loss_fn(logit, t, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            losses.append(loss.item())
        if len(va) == 0:  # full-data training: no validation set
            curve.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "sec": round(time.time() - t0, 1)})
            log(f"  ep {ep + 1:2d} loss {np.mean(losses):.4f} ({time.time() - t0:.0f}s)")
            continue
        val_logit = predict(model, imgs, va, mean, std, dev)
        auc = roc_auc_score(y[va], val_logit)
        msg = ""
        if ema:
            raw_auc = auc
            val_logit = predict(ema.model, imgs, va, mean, std, dev)
            auc = roc_auc_score(y[va], val_logit)
            msg = f" (raw {raw_auc:.4f})"
        curve.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_auc": float(auc),
                      "sec": round(time.time() - t0, 1)})
        log(f"  ep {ep + 1:2d} loss {np.mean(losses):.4f} val AUC {auc:.4f}{msg} ({time.time() - t0:.0f}s)")
        kill = cfg.get("kill_below")
        if kill and ep + 1 == kill[0] and auc < kill[1]:
            log(f"  killed: AUC {auc:.4f} < {kill[1]} at epoch {kill[0]}")
            break
    final = ema.model if ema else model
    res = {"val_logit": val_logit if len(va) else None, "curve": curve, "model": final}
    if len(va) == 0:
        return res
    if cfg.get("robustness", True):
        res["variant_logit"] = {v: predict(final, imgs, va, mean, std, dev, variant=v) for v in PERTURBATIONS}
    if cfg.get("tta_check"):  # flip-TTA ablation: same model, clean + every variant
        t_clean = roc_auc_score(y[va], predict(final, imgs, va, mean, std, dev, tta_flip=True))
        t_var = [roc_auc_score(y[va], predict(final, imgs, va, mean, std, dev, variant=v, tta_flip=True))
                 for v in PERTURBATIONS]
        n_var = [roc_auc_score(y[va], l) for l in res.get("variant_logit", {}).values()]
        log(f"  TTA check: clean {roc_auc_score(y[va], val_logit):.4f} -> {t_clean:.4f}; "
            f"robust mean {np.mean(n_var):.4f} -> {np.mean(t_var):.4f}; worst {np.min(n_var):.4f} -> {np.min(t_var):.4f}")
    return res
