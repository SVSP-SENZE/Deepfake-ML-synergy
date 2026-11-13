# Day 2 notes (Phase 2 start)

## Weights
ImageNet timm checkpoints, downloaded once as safetensors to `D:\deepfake\.cache\weights` and loaded from file
(`src/backbones.py`): convnext_tiny.fb_in22k_ft_in1k, efficientnet_b3.ra2_in1k,
vit_base_patch16_224.augreg2_in21k_ft_in1k, resnet50.a1_in1k. No CLIP / DINOv2 (rules question Q1 still open).

## Folds re-frozen with an embedding signal
- ConvNeXt-T last-layer features, standardised per dimension, cosine >= 0.75 added as a grouping edge.
- Catches re-crops and heavy degradations that pixel signals miss; over-merges a few similar scenes.
- Threshold chosen from review grids plus cluster sizes (0.70 -> max cluster 23; 0.75 -> max 11).
- 6,515 clusters, 909 images (13.0%) grouped, 101 clusters with both labels. Fold tests pass.

## Frozen-feature probe (T2.2), logistic regression, grouped 5-fold OOF
| Backbone | Last layer AUC | Intermediate layers AUC | Robust mean / worst (intermediate) |
|---|---|---|---|
| ViT-B/16 | 0.785 | **0.857** | 0.849 / 0.827 (blur 2.5) |
| ConvNeXt-T | 0.791 | 0.820 | 0.812 / 0.785 |
| ResNet-50 | 0.746 | 0.800 | 0.790 / 0.762 |
| EfficientNet-B3 | 0.748 | 0.791 | 0.782 / 0.755 |
- Intermediate layers beat the last layer for every backbone (+0.04 to +0.07). Adding last to intermediate does not help.
- C = 0.001 best on the grid 1e-4..1e-2 (C chosen on OOF: mild optimism, screening only).
- **Go / no-go gate: GO.** Semantic backbones carry strong signal.

## Shortcut check (coarse layout cue from Day 1)
Reference: logistic regression on 8x8 thumbnails, 0.686 AUC on current folds.
| Model | AUC | Spearman vs thumbnail | AUC within thumbnail-score quintiles |
|---|---|---|---|
| ViT-B probe (intermediate) | 0.857 | 0.465 | 0.826 |
| ConvNeXt-T fine-tune, fold 0 | 0.893 | 0.376 | 0.866 |
| ViT-B fine-tune, fold 0 | 0.903 | 0.369 | 0.881 |
Most of the signal is beyond the coarse cue; fine-tuned models depend on it less than probes.

## Fine-tune baseline (T2.3), fold 0 only, full fine-tune, hflip + 8px shift, AdamW, cosine, AMP
| Model | LR | Epochs | Fold-0 AUC (last epoch) | Peak epoch AUC | Robust mean / worst | s/epoch |
|---|---|---|---|---|---|---|
| ConvNeXt-T | 1e-4 | 10 | 0.893 | 0.907 (ep 3) | 0.890 / 0.875 (blur 2.5) | 25 |
| ViT-B/16 | 5e-5 | 8 | 0.903 | 0.910 (ep 2) | 0.898 / 0.880 (blur 2.5) | 50 |
- Both overfit after epoch 2–3 (train loss -> ~0.01). Next: shorter schedules, stronger augmentation,
  tuning regimes (LayerNorm-only, partial unfreeze), label smoothing.
- Strong blur is the consistent worst case -> blur augmentation in T3.1.
- Peak-epoch numbers are selected on validation and are optimistic; reported numbers use the last epoch.
