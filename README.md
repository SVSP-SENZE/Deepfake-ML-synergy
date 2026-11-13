# Deepfake detection: ViT-B/16 + ConvNeXt-T ensemble

IIIT Bangalore Deepfake ML Hackathon. Binary real/fake classifier for 224×224 face images.
**Grouped 5-fold out-of-fold AUC 0.909**, mean AUC over 11 image perturbations 0.905.
Full write-up: [`reports/Technical_Report.pdf`](reports/Technical_Report.pdf).

## Predict (what the judges run)

```bash
pip install -r requirements.txt          # Python 3.10; GPU build of torch, see first line of the file
python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv
```

- Output: `image_id,label,confidence`. `image_id` is the file name without extension,
  `confidence` is P(fake), and `label = confidence >= 0.5`.
- `model_weights.pt` (682 MB) holds all 6 models plus the ensemble config. No internet access needed.
- Speed: ~3 min for 3,000 images on an RTX 4060 Laptop GPU (fp16 autocast by default; `--fp32` is available).
- Two runs give byte-identical CSVs. `predictions.csv` in this folder was produced by the command above.
- Optional flags: `--batch_size` (default 64), `--threshold` (default 0.5), `--fp32`.

## Model

| Member | Pretraining (timm) | Fine-tuning | Seeds | Ensemble weight |
|---|---|---|---|---|
| ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | full, lr 5e-5, strong degradation aug | 3 | 0.7 |
| ConvNeXt-T | `convnext_tiny.fb_in22k_ft_in1k` | full, lr 1e-4, extra-strong degradation aug | 3 | 0.3 |

Per member, logits are averaged over seeds and horizontal-flip TTA, then standardised with constants taken from
5-fold out-of-fold training predictions and stored in the weights file. The members are combined as a weighted
sum, and confidence = sigmoid of that sum. Training used all 7,000 provided images except 677 flagged as likely
mislabelled by confident learning.

## Repository layout

```
inference_script.py      submission script (official template + fixes)
model_weights.pt         packed ensemble (scripts/pack_weights.py)
predictions.csv          submitted predictions for test/images (3,000 rows)
requirements.txt         pinned packages (requirements-lock.txt = full pip freeze)
src/                     data loading, backbones, augmentation, training loop, metrics, robustness proxy
scripts/                 every experiment and figure (see below)
tests/test_folds.py      asserts no near-duplicate cluster spans two folds
data/                    clusters.csv, folds.csv, label_issues.csv, duplicate candidates
experiments/results.csv  one row per experiment (46 runs): config, seed, AUC, robustness
experiments/runs/<id>/   config.json, train.log, out-of-fold logits per run
reports/                 Technical_Report.pdf (+ report.html), day-by-day notes D1–D8, figures/
weights/                 unpacked final models + ensemble.json
```

## Reproduce training

The data folder defaults to `../hackathon_dataset` (`train/real`, `train/fake`, `train_metadata.csv`). Set
`DATA_ROOT` to change it. The ImageNet checkpoints are read from `../.cache/weights/<arch>.<tag>.safetensors`,
downloaded once from `https://huggingface.co/timm/<arch>.<tag>/resolve/main/model.safetensors`.

```bash
# 1. near-duplicate clusters -> grouped folds (and check them)
python scripts/near_dups.py
python scripts/extract_features.py convnext_t --clean-only
python scripts/embed_dups.py
python scripts/make_folds.py
pytest tests/test_folds.py

# 2. 5-fold out-of-fold models used for label-noise estimation, then confident learning
python scripts/finetune_cv.py --backbone vit_b16 --lr 5e-5 --epochs 4 --folds 0 1 2 3 4 --name d3_vit_full_5fold
python scripts/finetune_cv.py --backbone convnext_t --lr 1e-4 --epochs 4 --folds 0 1 2 3 4 --name d3_cnx_full_5fold
python scripts/label_noise.py                      # -> data/label_issues.csv

# 3. cross-validated final recipe (for the ensemble constants and the report numbers)
python scripts/finetune_cv.py --backbone vit_b16 --lr 5e-5 --epochs 4 --degrade strong --prune --folds 0 1 2 3 4 --name vit_final_cv
python scripts/finetune_cv.py --backbone convnext_t --lr 1e-4 --epochs 4 --degrade xstrong --prune --folds 0 1 2 3 4 --name cnx_final_cv
python scripts/ensemble_eval.py vit=vit_final_cv cnx=cnx_final_cv   # -> experiments/ensemble_stats.json

# 4. final models on all (kept) training images, then pack
python scripts/train_full.py                       # 2 members x 3 seeds -> weights/
python scripts/pack_weights.py                     # -> model_weights.pt

# 5. figures
python scripts/evidence_figure.py
python scripts/gradcam.py
```

Each step takes 5–40 minutes on an RTX 4060. Runs are seeded (folds 42, CV 42 + fold, final models 1000–1002);
GPU nondeterminism can still move AUCs by about ±0.002. The submitted runs used the names in `experiments/results.csv`
(for example, the final CV is split across `d5_*_prune` and `d6_*_prune_f234`).

## Rules compliance

- Only the provided training data. Pretrained weights are the two ImageNet timm checkpoints above; no
  deepfake-specific detectors and no CLIP / DINOv2.
- Test images were used only to run the final inference script, never for training, selection or analysis.
- No hardcoded predictions or paths.
