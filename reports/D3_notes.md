# Day 3 notes (T2.4 tuning regimes)

## Regime screen, fold 0, single seed, last-epoch AUC
| Backbone | Regime | Trainable | LR | Epochs | AUC | Robust mean / worst |
|---|---|---|---|---|---|---|
| ViT-B/16 | full | 85.8M | 2e-5 | 5 | **0.909** | 0.904 / 0.890 |
| ViT-B/16 | full | 85.8M | 5e-5 | 4 | 0.908 | 0.904 / 0.889 |
| ConvNeXt-T | full | 27.8M | 1e-4 | 4 | 0.905 | 0.902 / 0.886 |
| ConvNeXt-T | full | 27.8M | 5e-5 | 5 | 0.902 | 0.897 / 0.879 |
| ViT-B/16 | partial (blocks 8–11) | 28.4M | 5e-5 | 5 | 0.886 | 0.881 / 0.858 |
| ConvNeXt-T | partial (stage 3) | 15.5M | 1e-4 | 5 | 0.864 | 0.856 / 0.822 |
| ViT-B/16 | LayerNorm only | 0.04M | 5e-4 | killed at 5 | 0.843 | 0.837 / 0.804 |
| ConvNeXt-T | LayerNorm only | 0.02M | 1e-3 | killed at 5 | 0.846 | 0.837 / 0.791 |

- Full fine-tuning wins for both backbones. Fold-0 CI is about +/-0.018, so the two ViT full runs are tied.
- **LayerNorm-only tuning is rejected**: -0.06 AUC versus full fine-tuning. This contradicts the Gemini
  report's +0.08 to +0.15 claim (from cross-generator studies on other data). Kill rule fired at epoch 5.
- Short schedules (4–5 epochs) fixed the Day 2 overfitting: last epoch ~ best epoch.

## 5-fold OOF, full fine-tune (hflip + 8px shift, AdamW wd 0.05, cosine, 1 warm-up epoch, AMP)
| Model | OOF AUC | 95% cluster CI | Per fold | Robust mean / worst |
|---|---|---|---|---|
| ViT-B/16, lr 5e-5, 4 ep | 0.898 | 0.890–0.907 | .906 .893 .899 .899 .892 | 0.894 / 0.875 (blur 2.5) |
| ConvNeXt-T, lr 1e-4, 4 ep | 0.892 | 0.885–0.901 | .905 .886 .891 .894 .887 | 0.889 / 0.873 (blur 2.5) |
| Mean of standardised logits | **0.901** | 0.894–0.911 | .912 .895 .902 .902 .896 | not measured yet |
| Mean of probabilities | 0.901 | 0.894–0.911 | same | — |

- The two-model average beats ViT alone on every fold (+0.004), but the members are highly correlated
  (Spearman 0.87), so the gain is small. Logit vs probability averaging: no difference.
- Rubric proxy at 0.90 AUC: ~40/50 accuracy points.
- From now on `finetune_cv.py` also saves perturbed-variant OOF logits, so ensembles get a robustness score.

## Next (Phase 3)
- Strong blur is still the worst case for every model -> degradation augmentation sweep (T3.1).
- Label noise handling (T3.3) using these OOF predictions.
