# Day 1 notes (Phase 0–1)

## Data structure
- Training set is built from source photos with heavy synthetic degradations applied to both classes:
  Gaussian/salt noise, grayscale, blur, pixelation, solarisation/tone curves, colour casts, grey/black/coloured patches.
  The same source scene often appears 2–6 times in different degraded versions.
- Some same-scene groups contain both labels (74 of 370 multi-image clusters). Visually these look like a real photo
  and a manipulated version of the same scene, not obvious label noise. To inspect properly in T3.4.

## Near-duplicate clusters (T1.2)
- Signals: c16 = correlation of z-scored 16x16 grayscale (max over mirror), pHash Hamming, c64 (64x64 grayscale).
- Calibrated by eye on grids in `reports/figures/dups/`:
  - c16 >= 0.93: same scene in ~all sampled pairs.
  - 0.90 <= c16 < 0.93 and pHash <= 8: same scene in ~10/12.
  - 0.90 <= c16 < 0.93 and pHash > 8: mostly different people (rejected).
  - pHash 9–12 alone: typical distance between any two aligned faces (meaningless).
- Rule: edge if c16 >= 0.93 or (c16 >= 0.90 and pHash <= 8). Connected components -> 6,613 clusters,
  757 images (10.8%) in multi-image clusters; sizes {1: 6243, 2: 356, 3: 13, 6: 1}. 79 opposite-label edges.
- **Known gap:** heavily degraded siblings (pixelation, solarisation, strong noise) are missed by pixel signals.
  Add an embedding-similarity signal when backbone weights are available (Day 2), then re-freeze folds.

## Folds (T1.3)
- 5-fold StratifiedGroupKFold on cluster_id, seed 42 -> `data/folds.csv`. Fold sizes 1,395–1,410, fake rate 0.49–0.52.
- `tests/test_folds.py`: coverage, no cluster in two folds, balance, labels match metadata. 4/4 pass.

## Baseline + control (D1 exit condition)
| Run | Grouped OOF AUC | 95% cluster-bootstrap CI | Robust mean / worst |
|---|---|---|---|
| LR on 32x32 RGB pixels + 7 stats | 0.707 | 0.694–0.720 | 0.707 / 0.701 (occlusion) |
| Same, shuffled train labels | ~0.495 | — | ~0.49 — control PASSED |

## Where the pixel signal comes from (scripts/pixel_signal_map.py)
| Features | Grouped AUC |
|---|---|
| RGB 32x32 | 0.705 |
| Gray 32x32 | 0.704 |
| Per-image normalised RGB | 0.703 |
| Centre 16x16 only / border only | 0.687 / 0.688 |
| RGB 8x8 | 0.687 |
- Colour adds nothing, and brightness/contrast normalisation does not remove it. Even 8x8 thumbnails carry 0.69 AUC.
- Mean-image difference: FAKE faces are darker inside a sharply bounded oval, with a brighter surround and a
  dark band at the hairline. Consistent with a face-swap blend mask, but could also be a source/alignment bias
  between the real and fake pools. The README says test fakes come from *different* generation methods, so
  this coarse cue may not transfer. Treat it as a risk to check on Day 2 (e.g. does a CNN still work when the
  coarse layout is suppressed?).

## Environment
- `.venv` Python 3.10.20, torch 2.7.1+cu126 (CUDA OK on RTX 4060), timm 1.0.15. Full list in `requirements-lock.txt`.
- Machine has ~3–4 GB of usable memory (C: drive full, page file can't grow). Run jobs one at a time;
  images are read from the on-disk cache `experiments/cache/train_u8.npy` (memory-mapped).
- Official `inference_template.py` has a SyntaxError (`global` after use); do not copy it as-is.
