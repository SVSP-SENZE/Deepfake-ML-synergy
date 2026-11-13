#!/usr/bin/env bash
# Day 6 / T4.1: third member screen + complete 5-fold OOF for the pruned recipes.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
f() { $PY scripts/finetune_cv.py --regime full --epochs 4 --prune "$@" 2>&1 | grep -v -i warn; }
f --backbone effnet_b3 --lr 3e-4 --degrade strong --folds 0 1 --name d6_eff_prune_f01 --hypothesis "T4.1 EffNet-B3 member screen"
f --backbone vit_b16 --lr 5e-5 --degrade strong --folds 2 3 4 --name d6_vit_prune_f234 --hypothesis "T4.1 ViT pruned, remaining folds"
f --backbone convnext_t --lr 1e-4 --degrade xstrong --folds 2 3 4 --name d6_cnx_prune_f234 --hypothesis "T4.1 ConvNeXt pruned, remaining folds"
echo D6 RUNS DONE
