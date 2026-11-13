"""T2.1 backbone wrapper: ImageNet timm models, weights loaded from local files only.

features(model, x) -> (last, mid)
  last: pre-logits pooled feature of the final layer
  mid:  concatenation of global-average-pooled intermediate feature maps
"""
from pathlib import Path

import numpy as np
import timm
import torch

from .common import ROOT

WEIGHTS_DIR = ROOT.parent / ".cache" / "weights"

# key: (timm architecture, pretrained tag, intermediate indices)
BACKBONES = {
    "convnext_t": ("convnext_tiny", "fb_in22k_ft_in1k", 4),
    "effnet_b3": ("efficientnet_b3", "ra2_in1k", 4),
    "vit_b16": ("vit_base_patch16_224", "augreg2_in21k_ft_in1k", [2, 5, 8, 11]),
    "resnet50": ("resnet50", "a1_in1k", 4),
}


def build(key: str, pretrained: bool = True, num_classes: int = 0) -> torch.nn.Module:
    arch, tag, _ = BACKBONES[key]
    kw = {}
    if pretrained:
        f = WEIGHTS_DIR / f"{arch}.{tag}.safetensors"
        assert f.exists(), f"missing local weights {f}"
        kw["pretrained_cfg_overlay"] = {"file": str(f)}
    return timm.create_model(f"{arch}.{tag}", pretrained=pretrained, num_classes=num_classes, **kw)


def norm_stats(model) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    return mean, std


def to_input(batch_u8: np.ndarray, mean, std, device) -> torch.Tensor:
    """uint8 [B,H,W,3] -> normalised float [B,3,H,W] on device."""
    x = torch.from_numpy(np.ascontiguousarray(batch_u8)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    return (x - mean.to(device)) / std.to(device)


@torch.inference_mode()
def features(model, key: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = BACKBONES[key][2]
    final, inter = model.forward_intermediates(x, indices=idx, output_fmt="NCHW")
    last = model.forward_head(final, pre_logits=True)
    mid = torch.cat([t.float().mean(dim=(2, 3)) for t in inter], dim=1)
    return last.float(), mid
