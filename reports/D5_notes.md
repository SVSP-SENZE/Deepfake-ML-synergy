# Day 5 notes (T3.3 label noise, T3.5 EMA)

## Confident learning (scripts/label_noise.py)
- Input: mean P(fake) of the Day 3 5-fold out-of-fold predictions (ViT-B + ConvNeXt-T).
- cleanlab estimates ~9.7% label noise, balanced (real->fake 0.047, fake->real 0.050).
- 677 images flagged (330 given REAL, 347 given FAKE). Review grids: reports/figures/label_issues_given_{real,fake}.png.
  The flags cannot be confirmed by eye (face swaps are subtle at 224 px), so part of this is "hard", not "wrong".
- Flag rate is 21.5% inside mixed-label clusters vs 9.2% elsewhere.

## Mixed-label clusters are not label noise
- 101 clusters hold both labels. Inspected pairs show the same scene with a subtly different face:
  a real photo and its face-swapped fake.
- The model ranks the fake-labelled member higher in 77.8% of 135 opposite-label pairs, so it already uses the
  face difference. Labels kept; no pair loss added. Grouping keeps each pair in one fold, so there's no leakage.

## Screen: ConvNeXt-T, xstrong augmentation, folds 0-1, 4 epochs (two-fold CI about ±0.012)
| Variant | Clean AUC | Seen mean | Unseen mean | Worst (pixelate) |
|---|---|---|---|---|
| baseline BCE (Day 4) | 0.900 | 0.900 | 0.893 | 0.871 |
| label smoothing 0.1 | 0.899 | 0.900 | 0.892 | 0.869 |
| GCE q=0.7 | 0.903 | 0.903 | 0.895 | 0.875 |
| SCE (beta 1) | 0.902 | 0.902 | 0.895 | 0.875 |
| **prune flagged (train folds only)** | **0.905** | **0.905** | **0.899** | **0.880** |
| EMA 0.99 | 0.901 | 0.901 | 0.893 | 0.870 |

- Pruning is best on all four metrics, and most of the gain is on fold 1 (0.893 -> 0.902). GCE/SCE come second.
- Label smoothing and EMA do nothing at 4 epochs (EMA tracks the raw weights closely; the cosine schedule already
  ends at a low learning rate).
- Caveat: the flags come from OOF predictions of models trained on all other folds, so fold f's labels slightly
  influenced which training images were dropped for fold f. The effect is mild (it only changes which training
  images are dropped; validation images are never removed), but it may inflate the gain a little.

## ViT-B confirmation
ViT-B/16, strong augmentation, lr 5e-5, folds 0-1:
| Variant | Clean AUC | Seen mean | Unseen mean | Worst (pixelate) | Folds |
|---|---|---|---|---|---|
| baseline BCE (Day 4) | 0.907 | 0.905 | 0.899 | 0.871 | 0.918 / 0.898 |
| prune flagged | 0.908 | 0.907 | 0.902 | 0.882 | 0.911 / 0.905 |

- Clean AUC is unchanged; robustness improves (unseen +0.003, worst case +0.011). Fold 1 improves on both backbones,
  and ViT fold 0 drops a little, so the clean gain is noise-level, but the worst-case gain repeats on both models.

## Decision
- **Adopt pruning of the cleanlab-flagged images for both members.** There's no clean cost, and robustness improved
  consistently on both backbones (worst case +0.010 / +0.011).
- GCE/SCE were a close second on ConvNeXt; not stacked with pruning (it would be a second fix for the same noise,
  with no time to test the combination properly).
- Label smoothing and EMA are rejected (no effect at 4 epochs).
- For the final full-data training, the flags are reused as-is (no validation fold there, so the leakage caveat
  doesn't apply).
