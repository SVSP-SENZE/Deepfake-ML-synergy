# Day 7 notes (T4.4 final training)

## What was trained
- Script: `scripts/train_full.py` (recipes frozen in the file). Output: `weights/` (651 MB total, < 5 GB limit).
- Data: all 7000 training images minus the 677 cleanlab-flagged ones = 6323 images. No validation hold-out.
- Members, 3 seeds each (seeds 1000-1002, distinct from the CV seeds):
  - ViT-B/16 (augreg2 in21k->in1k), full fine-tune, lr 5e-5, head lr x10, strong degradation aug. 3 x 172 MB.
  - ConvNeXt-T (fb in22k->in1k), full fine-tune, lr 1e-4, head lr x10, xstrong degradation aug. 3 x 56 MB.
  - Shared: 4 epochs, AdamW wd 0.05, batch 64, cosine with 1 warm-up epoch, AMP fp16, hflip + 8 px shift, BCE.
- Saved as fp16 state dicts. Training time: ~5-6 min per model on the RTX 4060 (~35 min total).

## ensemble.json (read by the inference script; nothing is hardcoded there)
- Per member: timm name, normalisation mean/std, weight file list, ensemble weight (ViT 0.7, ConvNeXt 0.3),
  z-logit mean/std from the 5-fold OOF logits (Day 6).
- Combination: per member, average logits over seeds and over flip TTA; standardise with the stored OOF stats;
  weighted sum; confidence = sigmoid(sum).

## Sanity checks (on 600 training images + the 677 pruned images)
- All 6 files load with `strict=True` into `timm.create_model(..., pretrained=False, num_classes=1)`, so
  inference needs no ImageNet download.
- Logit spread of the final models (std 5.5-7.5) matches the OOF stats used for standardisation (6.7 / 5.4), so the
  0.7/0.3 weighting behaves as tested.
- Training images: AUC ~1.00 (expected; they were trained on).
- Pruned images (never trained on): the models agree with the given label only ~8% of the time. This matches
  cleanlab's reason for flagging them (out-of-fold models also disagreed), so it's consistent, not a bug.
- Seed agreement (Spearman): ViT 0.96-0.97, ConvNeXt 0.98. Seeds differ enough to give some averaging benefit.

## Caveats
- No held-out score exists for the final models; the expected performance is the 5-fold OOF figure
  (ensemble ~0.909 clean, 0.905 robust mean). Full-data training and seed averaging should be equal or slightly better.
- The z-logit stats come from single-seed fold models. Averaging 3 seeds narrows the logit spread a little, so the
  effective ViT/ConvNeXt balance shifts slightly. AUC is insensitive to this within the flat 0.6-0.8 range seen on Day 6.
