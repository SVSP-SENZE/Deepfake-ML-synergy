#!/usr/bin/env bash
# Day 4 / T3.1: degradation-augmentation strength sweep, ConvNeXt-T, folds 0-1, single seed.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
H="T3.1 degradation strength sweep (folds 0-1)"
run() { $PY scripts/finetune_cv.py --backbone convnext_t --regime full --lr 1e-4 --folds 0 1 \
        --hypothesis "$H" "$@" 2>&1 | grep -v -i warn; }

run --degrade off    --epochs 4 --name d4_cnx_deg_off
run --degrade weak   --epochs 4 --name d4_cnx_deg_weak
run --degrade medium --epochs 4 --name d4_cnx_deg_medium
run --degrade strong --epochs 4 --name d4_cnx_deg_strong
run --degrade strong --epochs 6 --name d4_cnx_deg_strong_e6
echo SWEEP DONE
