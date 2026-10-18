"""Shared metrics for every model (XGBoost baseline and all GNNs)."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)


def best_threshold(y_true, prob):
    """Pick the probability threshold that maximizes illicit-class F1 on VALIDATION."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, prob >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(y_true, prob, thr):
    """Metrics for the illicit (positive) class."""
    pred = prob >= thr
    return {
        "threshold": thr,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def per_step_f1(y_true, prob, steps, thr):
    """Illicit-class F1 per time step (None when a step has no illicit nodes)."""
    out = {}
    for s in np.unique(steps):
        m = steps == s
        if y_true[m].sum() == 0:
            out[int(s)] = None
        else:
            out[int(s)] = float(f1_score(y_true[m], prob[m] >= thr, zero_division=0))
    return out
