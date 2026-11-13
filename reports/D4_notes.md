# Day 4 notes (T3.1 degradation augmentation)

## Setup
- `src/augment.py`: colour jitter -> down-up resample -> Gaussian blur -> JPEG, each with probability p,
  same for both classes, on CPU before GPU upload (~0.1 s per batch of 64).
- Levels: weak p=0.2 (JPEG 70–95, sigma 0.5–1.2, scale 0.7–0.9) · medium p=0.3 (50–95, 0.5–2.0, 0.5–0.9) ·
  strong p=0.5 (30–95, 0.5–2.5, 0.5–0.9) · xstrong p=0.7 (20–95, 0.5–3.0, 0.35–0.9). Colour ±10–25%.
- Robustness proxy now has 11 variants, split into **seen** families (JPEG 75/50, blur 1/2.5, down-up 0.5,
  colour) and **unseen** families never used in training (noise, occlusion, grayscale, pixelation x0.25,
  gamma 0.6). The unseen score is the honest generalisation check.
- Folds 0–1, single seed, 4 epochs unless stated. CI on two folds is about ±0.012.

## ConvNeXt-T (lr 1e-4)
| Strength | Clean AUC | Seen mean | Unseen mean | Worst (pixelate) | Blur 2.5 |
|---|---|---|---|---|---|
| off | 0.895 | 0.892 | 0.878 | 0.817 | 0.880 |
| weak | 0.900 | 0.896 | 0.883 | 0.827 | 0.885 |
| medium | 0.899 | 0.897 | 0.881 | 0.821 | 0.890 |
| strong | **0.901** | 0.899 | 0.889 | 0.848 | 0.894 |
| strong, 6 epochs | 0.896 | 0.895 | 0.883 | 0.839 | 0.888 |
| **xstrong** | 0.900 | **0.900** | **0.893** | **0.871** | **0.897** |

## ViT-B/16 (lr 5e-5)
| Strength | Clean AUC | Seen mean | Unseen mean | Worst (pixelate) | Blur 2.5 |
|---|---|---|---|---|---|
| off | 0.900 | 0.896 | 0.887 | 0.848 | 0.879 |
| **strong** | **0.907** | **0.905** | **0.899** | 0.871 | **0.898** |
| xstrong | 0.899 | 0.899 | 0.895 | **0.878** | 0.895 |

## Decision
- **ViT-B: strong. ConvNeXt-T: xstrong.** Both chosen on clean AUC and robustness jointly; neither costs clean AUC
  versus no augmentation (the build map's kill criterion was a clean loss > 0.03).
- Augmentation helps unseen damage types too (+0.011 to +0.015), so it is not just memorising the test perturbations.
- Longer training with strong augmentation did not help (6 epochs < 4 epochs).
- Pixelation (unseen) is now the worst case for every model. Not added to training, to keep an honest unseen check;
  revisit if the final robustness score needs it.
- Different strengths for the two members also add a little diversity for the ensemble.
