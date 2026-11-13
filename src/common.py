"""Shared paths, seeding and the experiment log."""
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Keep every download and cache on the project drive (C: is full).
_CACHE_HOME = ROOT.parent / ".cache"
for _k, _sub in [("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"),
                 ("XDG_CACHE_HOME", ""), ("MPLCONFIGDIR", "matplotlib")]:
    os.environ.setdefault(_k, str(_CACHE_HOME / _sub))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT.parent / "hackathon_dataset"))
CACHE = ROOT / "experiments" / "cache"
RUNS = ROOT / "experiments" / "runs"
RESULTS_CSV = ROOT / "experiments" / "results.csv"
FOLDS_CSV = ROOT / "data" / "folds.csv"
CLUSTERS_CSV = ROOT / "data" / "clusters.csv"

RESULT_FIELDS = ["time", "run_id", "hypothesis", "config", "seed", "git",
                 "cv_auc", "cv_auc_std", "robust_mean_auc", "robust_worst_auc",
                 "runtime_s", "notes"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def git_hash() -> str:
    """Short hash if the project is its own git repo, else 'no-git'."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
        if Path(top).resolve() != ROOT.resolve():
            return "no-git"
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "no-commit"
    except OSError:
        return "no-git"


def start_run(name: str, config: dict, seed: int) -> Path:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {"run_id": run_id, "seed": seed, "git": git_hash(), **config}, indent=2))
    return d


def log_result(run_dir: Path, hypothesis: str, config: dict, seed: int, **metrics) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    row = {k: "" for k in RESULT_FIELDS}
    row.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), run_id=run_dir.name,
               hypothesis=hypothesis, config=json.dumps(config, sort_keys=True),
               seed=seed, git=git_hash())
    row.update({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in metrics.items()})
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
