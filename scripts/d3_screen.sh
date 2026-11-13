#!/usr/bin/env bash
# Day 3 / T2.4: tuning-regime screen on fold 0 (single seed). Runs sequentially (low host memory).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
H="T2.4 regime screen (fold 0)"
run() { $PY scripts/finetune_cv.py --folds 0 --hypothesis "$H" "$@" 2>&1 | grep -v -i warn; }

run --backbone vit_b16    --regime full      --lr 5e-5 --epochs 4 --name d3_vit_full_lr5e-5_e4
run --backbone vit_b16    --regime full      --lr 2e-5 --epochs 5 --name d3_vit_full_lr2e-5_e5
run --backbone vit_b16    --regime layernorm --lr 5e-4 --epochs 8 --name d3_vit_ln_lr5e-4_e8 --kill_below 5 0.86
run --backbone vit_b16    --regime partial   --lr 5e-5 --epochs 5 --name d3_vit_partial_lr5e-5_e5
run --backbone convnext_t --regime full      --lr 1e-4 --epochs 4 --name d3_cnx_full_lr1e-4_e4
run --backbone convnext_t --regime full      --lr 5e-5 --epochs 5 --name d3_cnx_full_lr5e-5_e5
run --backbone convnext_t --regime layernorm --lr 1e-3 --epochs 8 --name d3_cnx_ln_lr1e-3_e8 --kill_below 5 0.85
run --backbone convnext_t --regime partial   --lr 1e-4 --epochs 5 --name d3_cnx_partial_lr1e-4_e5
echo SCREEN DONE
