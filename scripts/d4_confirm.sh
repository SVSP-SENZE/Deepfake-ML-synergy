#!/usr/bin/env bash
# Day 4: confirm strength on ViT-B (folds 0-1), and probe one stronger level on ConvNeXt-T.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
H="T3.1 degradation strength (folds 0-1)"
run() { $PY scripts/finetune_cv.py --regime full --folds 0 1 --epochs 4 --hypothesis "$H" "$@" 2>&1 | grep -v -i warn; }
run --backbone convnext_t --lr 1e-4 --degrade xstrong --name d4_cnx_deg_xstrong
run --backbone vit_b16    --lr 5e-5 --degrade off     --name d4_vit_deg_off
run --backbone vit_b16    --lr 5e-5 --degrade strong  --name d4_vit_deg_strong
echo CONFIRM DONE
