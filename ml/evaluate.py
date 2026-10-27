"""Metrics that stay meaningful when attacks are rare."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def confusion(y, score, thr):
    pred = score >= thr
    y = np.asarray(y).astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return tn, fp, fn, tp


def metrics_at(y, score, thr) -> dict:
    tn, fp, fn, tp = confusion(y, score, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def threshold_for_fpr(y, score, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate on `y` is <= target_fpr."""
    benign = np.sort(np.asarray(score)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(target_fpr * len(benign)))
    k = min(k, len(benign) - 1)
    return float(np.nextafter(benign[k], np.inf))


def recall_at_fpr(y, score, target_fpr: float) -> float:
    return metrics_at(y, score, threshold_for_fpr(y, score, target_fpr))["recall"]


def threshold_free(y, score, target_fpr: float) -> dict:
    return {"pr_auc": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            f"recall_at_{target_fpr:.0%}_fpr": recall_at_fpr(y, score, target_fpr)}


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Exact precision if the same detector saw traffic with a different attack rate."""
    tp, fp = tpr * prevalence, fpr * (1 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0


def curves(y, score, n_points: int = 81) -> dict:
    """Compact PR and ROC curves for the dashboard.

    PR uses interpolated precision (best precision achievable at >= each recall level),
    sampled on an even recall grid so the knee of the curve is not under-sampled.
    """
    p, r, _ = precision_recall_curve(y, score)
    grid = np.linspace(0, 1, n_points)
    prec = [float(p[r >= g].max()) if (r >= g).any() else 0.0 for g in grid]
    fpr, tpr, _ = roc_curve(y, score)
    fgrid = np.unique(np.r_[np.linspace(0, 0.05, 51), np.linspace(0.05, 1, 40)])
    return {"pr": {"recall": [round(float(g), 4) for g in grid], "precision": [round(v, 5) for v in prec]},
            "roc": {"fpr": [round(float(g), 4) for g in fgrid],
                    "tpr": [round(float(v), 5) for v in np.interp(fgrid, fpr, tpr)]}}
