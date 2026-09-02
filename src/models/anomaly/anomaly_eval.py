"""
Shared anomaly-model evaluation utilities
=========================================
Scoring helper for the three anomaly models (Isolation Forest, feedforward AE, LSTM AE

Scoring convention:
    HIGHER anomaly score = MORE anomalous = more likely PHISHING.

Label convention:
    0 = phishing, 1 = legitimate
Evaluates the model's ability to give phishing (label 0) HIGHER anomaly scores than legitimate (1).
therefore converting to a detection target y_phish = 1 where label == 0.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


def to_phishing_target(labels):
    """G2 v2 uses 0=phishing, 1=legit. Detection target = 1 for phishing."""
    labels = np.asarray(labels).astype(int)
    return (labels == 0).astype(int)


def pick_threshold(scores, y_phish, strategy="youden", target_fpr=0.10):
    """Choose an anomaly-score threshold on a validation set.

    strategy:
      'f1'     -> maximise F1
      'youden' -> maximise Youden's J = TPR - FPR (less imbalance-biased)
      'fpr'    -> lowest threshold whose FPR <= target_fpr (falls back to youden)
    Returns (threshold, achieved_strategy_value).
    """
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best_t, best_val = float(candidates[0]), -1.0

    if strategy == "fpr":
        feasible = []
        for t in candidates:
            pred = (scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_phish, pred, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            if fpr <= target_fpr:
                feasible.append((tpr, t))
        if feasible:
            feasible.sort(reverse=True)
            return float(feasible[0][1]), float(feasible[0][0])
        strategy = "youden"

    for t in candidates:
        pred = (scores >= t).astype(int)
        if strategy == "youden":
            tn, fp, fn, tp = confusion_matrix(y_phish, pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            val = tpr - fpr
        else:
            val = f1_score(y_phish, pred, zero_division=0)
        if val > best_val:
            best_val, best_t = val, float(t)
    return best_t, float(best_val)


def evaluate(scores, y_phish, threshold):
    """Full metric suite at a given threshold. y_phish: 1=phishing, 0=legit."""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_phish, pred, labels=[0, 1]).ravel()
    return {
        "precision": round(precision_score(y_phish, pred, zero_division=0), 4),
        "recall": round(recall_score(y_phish, pred, zero_division=0), 4),
        "f1": round(f1_score(y_phish, pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_phish, scores), 4),
        "auc_pr": round(average_precision_score(y_phish, scores), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_all_strategies(val_scores, y_val_phish, test_scores, y_test_phish,
                            target_fpr=0.10):
    """Pick a threshold on val for each strategy, report test metrics for each."""
    results = {}
    for strat in ["f1", "youden", "fpr"]:
        thr, sval = pick_threshold(val_scores, y_val_phish, strat, target_fpr)
        m = evaluate(test_scores, y_test_phish, thr)
        m["threshold"] = round(thr, 6)
        m["strategy_value"] = round(sval, 4)
        results[strat] = m
    return results
