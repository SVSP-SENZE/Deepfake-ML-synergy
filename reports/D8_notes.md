# Day 8 notes (T5: inference script, timing, reproduction)

## Deliverables
- `inference_script.py`: built on the official template, same CLI
  (`--test_dir --weights --output`, plus optional `--batch_size --threshold --fp32`).
  - Fixed: the template's `global BATCH_SIZE, THRESHOLD` after use is a SyntaxError; values are now passed as arguments.
  - Image ID = filename stem; preprocessing identical to training (RGB, 224x224, PIL bilinear only if the size differs).
  - Loads everything from the single weights file; needs `timm` but no internet (pretrained=False).
- `model_weights.pt` (682 MB): 6 fp16 state dicts + ensemble config, built by `scripts/pack_weights.py`.
- `predictions.csv`: 3000 rows, `image_id,label,confidence`, IDs match sample_submission.csv exactly.

## Checks
| Check | Result |
|---|---|
| Script vs training-code path (128 training images) | max confidence diff 0.0001, labels 100% equal |
| fp16 (default) vs fp32 (3478 images) | labels agree 99.97%, max confidence diff 0.0005 |
| Two full runs on the test folder | byte-identical CSV |
| Threshold 0.5 on 5-fold OOF | best accuracy (0.874) of 0.40-0.60; predicts 50.2% fake vs 50.3% true |
| Test predictions | 49.9% predicted fake, mean confidence 0.494 |
| Confidence on pruned (likely mislabelled) training reals | 9.1% called fake, matching the share flagged by cleanlab |

## Timing (RTX 4060 Laptop, thermally throttled: 1–1.9 GHz of 3.1 GHz, 75–84 °C)
| Setting | Images | Time |
|---|---|---|
| fp32, TTA, 6 models | 3478 | 240 s |
| fp16, TTA, 6 models (default) | 3000 | 158–165 s script, 183–190 s incl. Python start |
| breakdown (fp16) | | import ~25 s, model load ~12 s, data loading negligible, rest GPU |

- Under the 5-minute (20-point) tier on this machine with ~1.6x margin. **Not yet measured on a real T4.**
  A T4 (65 TFLOPS fp16 tensor cores) should be similar or faster than this throttled laptop GPU, but this is unverified.
- Fallbacks if the T4 is over 5 min: 1 seed per member (~3x faster), or no flip TTA (2x faster; -0.002 AUC).

## T4 check (Colab / Kaggle, GPU T4)
```
!pip install -q timm==1.0.15
# upload inference_script.py, model_weights.pt and the test images folder, then:
!time python inference_script.py --test_dir test/images --weights model_weights.pt --output predictions_t4.csv
```
Compare `predictions_t4.csv` with `predictions.csv`: labels should agree >= 99.9%, confidence diff < 0.001.
