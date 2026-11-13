# Day 6 notes (T4.1 diverse members, T4.2 combination, flip TTA)

All members use the Day 5 recipe: full fine-tune, 4 epochs, degradation augmentation, cleanlab-flagged images
pruned from training folds. Out-of-fold (OOF) predictions now cover all 5 folds for ViT-B and ConvNeXt-T.

## Members (5-fold OOF, 7000 images)
| Member | Family | Clean AUC [cluster CI] | Unseen mean | Robust mean | Worst (pixelate) |
|---|---|---|---|---|---|
| ViT-B/16 (lr 5e-5, strong aug) | transformer | 0.908 [0.902-0.917] | 0.901 | 0.904 | 0.882 |
| ConvNeXt-T (lr 1e-4, xstrong aug) | CNN | 0.902 [0.896-0.911] | 0.894 | 0.898 | 0.870 |
| EfficientNet-B3 (lr 3e-4, folds 0-1 only) | CNN (BN) | 0.828 | 0.782 | 0.813 | 0.691 |

- Spearman of OOF logits: ViT-ConvNeXt 0.937; EfficientNet-others ~0.74 (diverse but much weaker).
- EfficientNet-B3 at 224 px was weak in the Day 2 probe too (0.79). Loss ~1.5 in epoch 1 suggests lr 3e-4 was too
  high; not re-tuned (time), so dropped.

## Combination (5-fold OOF)
| Ensemble | Clean | Unseen | Robust mean | Worst |
|---|---|---|---|---|
| ViT alone | 0.9078 | 0.9011 | 0.9037 | 0.8818 |
| ViT+ConvNeXt, z-logit 0.5/0.5 | 0.9080 | 0.9022 | 0.9047 | 0.8854 |
| ViT+ConvNeXt, probability mean | 0.9075 | 0.9014 | 0.9041 | 0.8837 |
| ViT+ConvNeXt, rank mean | 0.9080 | 0.9022 | 0.9047 | 0.8853 |
| **ViT+ConvNeXt, z-logit 0.7/0.3** | **0.9086** | **0.9028** | **0.9052** | **0.8861** |
| ViT+ConvNeXt+EffNet 0.45/0.45/0.1 (folds 0-1) | +0.0006 vs 2-member | +0.0002 | +0.0002 | -0.0009 |

- z-logit = each member's logit standardised by mean/std of its clean OOF logits (stored in
  experiments/ensemble_stats.json, shipped with the weights); nothing is fitted on test data.
- The weight curve is flat between 0.6 and 0.8 for ViT; 0.7/0.3 is picked on OOF, so it is mildly optimistic.
- Per fold, the ensemble beats ViT on folds 0-1 and is level or slightly behind on folds 2-4 (clean). The clean gain
  is within noise; the robustness gain (worst +0.004, unseen +0.002) is small but consistent.
- Why keep two members anyway: the hidden test uses unseen generation methods. A CNN and a transformer fail in
  different ways (Spearman 0.94, not 1), which is a hedge against that shift; it is also the +2 diversity bonus.
  The cost is about one extra forward pass.

## Flip TTA (fold 0, same trained model with and without)
| Member | Clean | Robust mean | Worst |
|---|---|---|---|
| ViT-B | 0.9111 -> 0.9129 | 0.9084 -> 0.9102 | 0.8862 -> 0.8893 |
| ConvNeXt-T | 0.9087 -> 0.9085 | 0.9060 -> 0.9060 | 0.8813 -> 0.8838 |

- Small, mostly positive; doubles inference cost, which stays far inside the 5-minute T4 budget. **Adopted.**

## Decision for Day 7 (final training)
- Ship **ViT-B/16 + ConvNeXt-T**, z-logit average, weights 0.7/0.3, flip TTA.
- Retrain each on all training data (pruned), same epochs and recipe; optionally 2-3 seeds per member if time and
  the T4 budget allow.
- Dropped: EfficientNet-B3, probability/rank averaging.
