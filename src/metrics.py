"""T1.5 metrics and rubric-point conversion."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def basic(y, p, threshold: float = 0.5) -> dict:
    yhat = (np.asarray(p) >= threshold).astype(int)
    return {"auc": float(roc_auc_score(y, p)), "acc": float(accuracy_score(y, yhat)),
            "f1": float(f1_score(y, yhat)), "threshold": threshold}


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1 (use on out-of-fold predictions only)."""
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    return float(max(cand, key=lambda t: f1_score(y, (p >= t).astype(int))))


def cluster_bootstrap_auc(y, p, clusters, n_boot: int = 1000, seed: int = 0) -> tuple:
    """95% CI of AUC, resampling whole near-duplicate clusters."""
    y, p, clusters = map(np.asarray, (y, p, clusters))
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def accuracy_points(auc: float) -> float:
    return max(0.0, (auc - 0.5) / 0.5) * 50


def efficiency_points(seconds: float) -> int:
    m = seconds / 60
    return 20 if m < 5 else 15 if m < 10 else 10 if m < 20 else 5 if m < 40 else 0


def robustness_points(variant_aucs: list[float]) -> float:
    """Proxy only: the real weighting is undisclosed. Mean variant AUC on the accuracy scale."""
    return max(0.0, (float(np.mean(variant_aucs)) - 0.5) / 0.5) * 20
