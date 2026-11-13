#!/usr/bin/env bash
# Day 5 / T3.3 + T3.5: label-noise handling and EMA. ConvNeXt-T, xstrong, folds 0-1, 4 epochs.
# Baseline for comparison: d4_cnx_deg_xstrong (BCE, no EMA).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
H="T3.3/T3.5 noise handling + EMA (folds 0-1)"
run() { $PY scripts/finetune_cv.py --backbone convnext_t --regime full --lr 1e-4 --folds 0 1 --epochs 4 \
        --degrade xstrong --hypothesis "$H" "$@" 2>&1 | grep -v -i warn; }
run --label_smoothing 0.1 --name d5_cnx_ls0.1
run --loss gce --gce_q 0.7 --name d5_cnx_gce
run --loss sce --name d5_cnx_sce
run --prune --name d5_cnx_prune
run --ema 0.99 --name d5_cnx_ema
echo D5 SCREEN DONE
